import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

from src.data import SMELL_CLASSES, SMELL_TO_IDX, SmellDataset, PATTERN_CLASSES, PATTERN_TO_IDX
from src.environment import SOFT_REWARD_MATRIX

logger = logging.getLogger("SmellRL.eval")

# Map ground-truth smells to their dominant refactoring pattern index:
# 0: GodClass -> 2: Facade
# 1: FeatureEnvy -> 0: Strategy
# 2: LongMethod -> 4: ExtractClass
# 3: DataClass -> 1: Observer
# 4: NoSmell -> 5: None
SMELL_TO_PATTERN_DOMINANT = {
    0: 2,
    1: 0,
    2: 4,
    3: 1,
    4: 5,
}

# ──────────────────────────── Metric helpers ───────────────────────────────

def classification_report_dict(y_true: List[int], y_pred: List[int], labels: List[str]) -> dict:
    """Compute per-class + macro precision/recall/F1/accuracy and confusion matrix."""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    n = len(labels)
    report = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {},
    }
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    report["confusion_matrix"] = cm.tolist()
    
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n)))
    r = recall_score(y_true, y_pred,    average=None, zero_division=0, labels=list(range(n)))
    f = f1_score(y_true, y_pred,        average=None, zero_division=0, labels=list(range(n)))
    for i, name in enumerate(labels):
        report["per_class"][name] = {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i])}
    return report

# ──────────────────────────── Baseline agents ──────────────────────────────

class RandomAgent:
    """Uniformly random refactoring actions — establishes floor performance."""
    name = "Random Agent"
    def __init__(self, n_actions: int = 6, seed: int = 42):
        self.n_a = n_actions
        np.random.seed(seed)

    def predict(self, _item: dict) -> int:
        return np.random.randint(0, self.n_a)

class RuleBasedAgent:
    """
    Designite-inspired rule-based smell detector adapted to Halstead metrics.
    Output mapped to smells (which matches true class).
    """
    name = "Rule-Based (Designite-like)"

    def predict(self, item: dict) -> int:
        # Features on class node x[0]
        x = item["x"]
        cf = x[0].tolist()

        # Reverse-normalise Halstead metrics
        cyclomatic = cf[3] * 50.0
        lloc = cf[4] * 5000.0
        V = cf[5] * 5000.0
        D = cf[6] * 100.0
        E = cf[7] * 1000000.0
        B = cf[8] * 5.0
        
        n_m = max(1, int(x[:, 1].sum().item()))
        n_f = max(0, int(x[:, 2].sum().item()))
        
        if cyclomatic > 30.0 or lloc > 1500.0:
            return SMELL_TO_IDX["GodClass"]
        if D > 50.0 and V > 2000.0:
            return SMELL_TO_IDX["FeatureEnvy"]
        if (lloc / n_m) > 100.0 or (cyclomatic / n_m) > 8.0:
            return SMELL_TO_IDX["LongMethod"]
        if n_f > 8 and cyclomatic < 10.0:
            return SMELL_TO_IDX["DataClass"]
        return SMELL_TO_IDX["NoSmell"]

class SVMAgent:
    """SVM trained on Halstead features to predict smells."""
    name = "SVM + Halstead Metrics"

    def __init__(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.clf = SVC(kernel="rbf", C=1.0, probability=False)
        self.scaler = StandardScaler()
        self.trained = False

    def _extract_features(self, item: dict) -> np.ndarray:
        x = item["x"][0].numpy()
        # Halstead metrics at indices 3:9
        return x[3:9]

    def fit(self, train_ds: SmellDataset):
        X = np.stack([self._extract_features(item) for item in train_ds])
        y = np.array([int(item["y"].item()) for item in train_ds])
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, item: dict) -> int:
        feat = self._extract_features(item).reshape(1, -1)
        feat = self.scaler.transform(feat)
        return int(self.clf.predict(feat)[0])

# ──────────────────────────── RL-Native Metrics ────────────────────────────

def compute_cumulative_reward(preds: List[int], test_ds: SmellDataset, class_weights: dict) -> float:
    """Measures soft reward accumulated by predictions on the test set."""
    total_reward = 0.0
    for item, action in zip(test_ds, preds):
        true_smell = int(item["y"].item())
        if true_smell < 0 or true_smell >= len(SOFT_REWARD_MATRIX):
            base = -2.0
        else:
            if action < 0 or action >= 6:
                base = -2.0
            else:
                base = SOFT_REWARD_MATRIX[true_smell][action]
        total_reward += base * class_weights.get(true_smell, 1.0)
    return total_reward / max(1, len(test_ds))

def compute_halstead_delta(preds: List[int], test_ds: SmellDataset) -> float:
    """
    Computes average simulated Halstead metric reduction on the test set.
    For correct actions, simulates metric reduction according to env transitions.
    For incorrect refactorings on NoSmell, simulates coupling overhead.
    """
    initial_sum = 0.0
    final_sum = 0.0
    
    # Action transitions definitions
    # FACADE=2, STRATEGY=0, EXTRACT=4, OBSERVER=1, MEDIATOR=3, NO_REFACTOR=5
    for item, action in zip(test_ds, preds):
        true_smell = int(item["y"].item())
        h = item["halstead"].numpy() # [cyclomatic, lloc, V, D, E, B]
        
        c_before = h[0] + h[1] + h[2] + h[3] + h[4] + h[5]
        initial_sum += c_before
        
        c_after = h[0]
        curr_cyclomatic = h[0]
        curr_lloc = h[1]
        curr_V = h[2]
        curr_D = h[3]
        curr_E = h[4]
        curr_B = h[5]
        
        if action == 2:      # God Class -> Facade
            curr_cyclomatic *= 0.725
            curr_lloc *= 0.80
            curr_V *= 0.80
            curr_E *= 0.80
            curr_B *= 0.80
        elif action == 0:    # FeatureEnvy -> Strategy
            curr_D *= 0.80
            curr_E *= 0.80
        elif action == 4:    # LongMethod -> ExtractClass
            curr_cyclomatic *= 0.65
            curr_lloc *= 0.65
            curr_V *= 0.65
            curr_E *= 0.65
            curr_B *= 0.65
        elif action == 1:    # DataClass -> Observer
            curr_D *= 0.825
            curr_E *= 0.825
        elif action == 3:    # Mediator
            curr_cyclomatic *= 0.80
            curr_D *= 0.875
            curr_E *= 0.875
        elif action == 5:    # NoRefactor
            pass
            
        # Unnecessary refactoring penalty on clean code
        if true_smell == 4 and action in [0, 1, 2, 3, 4]:
            curr_D *= 1.15
            curr_E *= 1.15
            
        c_after = curr_cyclomatic + curr_lloc + curr_V + curr_D + curr_E + curr_B
        final_sum += c_after
        
    delta_pct = (initial_sum - final_sum) / max(1e-9, initial_sum) * 100.0
    return float(delta_pct)

# ──────────────────────────── Experiment Runner ────────────────────────────

class ExperimentRunner:
    def __init__(self, agent: nn.Module, train_ds: SmellDataset, test_ds: SmellDataset, cfg: dict, device: torch.device):
        self.agent = agent
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.cfg = cfg
        self.device = device
        self.results_dir = cfg["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
        self.log = logger

        # Compute training class weights for reward metrics evaluation
        counts = {}
        for item in train_ds:
            y = int(item["y"].item())
            counts[y] = counts.get(y, 0) + 1
        n_classes = cfg["smells"]["n_classes"]
        total = sum(counts.values())
        self.class_weights = {
            cls_idx: total / (n_classes * count) if count > 0 else 1.0
            for cls_idx, count in counts.items()
        }

    @torch.no_grad()
    def _smellrl_preds(self, ds: SmellDataset) -> List[int]:
        self.agent.eval()
        preds = []
        for item in ds:
            action, _ = self.agent.select_action(item, epsilon=0.0)
            preds.append(action)
        self.agent.train()
        return preds

    def run_baseline_comparison(self) -> dict:
        self.log.info("[Eval] ─── Experiment: Baseline Comparison ───")
        true_y = [SMELL_TO_PATTERN_DOMINANT[int(item["y"].item())] for item in self.test_ds]
        methods = {}

        rand = RandomAgent()
        methods["Random"] = [rand.predict(it) for it in self.test_ds]

        rule = RuleBasedAgent()
        methods["Rule-Based"] = [SMELL_TO_PATTERN_DOMINANT[rule.predict(it)] for it in self.test_ds]

        svm = SVMAgent()
        svm.fit(self.train_ds)
        methods["SVM"] = [SMELL_TO_PATTERN_DOMINANT[svm.predict(it)] for it in self.test_ds]

        methods["SmellRL (Ours)"] = self._smellrl_preds(self.test_ds)

        results = {}
        rows = []
        for name, preds in methods.items():
            rep = classification_report_dict(true_y, preds, PATTERN_CLASSES)
            mean_rwd = compute_cumulative_reward(preds, self.test_ds, self.class_weights)
            halstead_delta = compute_halstead_delta(preds, self.test_ds)
            
            results[name] = rep
            results[name]["mean_reward"] = mean_rwd
            results[name]["halstead_delta"] = halstead_delta
            
            rows.append({
                "Method": name, 
                "Accuracy": round(rep["accuracy"], 4), 
                "F1": round(rep["f1"], 4), 
                "Precision": round(rep["precision"], 4), 
                "Recall": round(rep["recall"], 4),
                "Mean_Cumulative_Reward": round(mean_rwd, 4),
                "Mean_Halstead_Delta(%)": round(halstead_delta, 2)
            })
            self.log.info(f"[Eval] {name:15s} | Acc={rep['accuracy']:.4f} | F1={rep['f1']:.4f} | MeanRwd={mean_rwd:+.3f} | MetricDelta={halstead_delta:.1f}%")
            
            if name == "SmellRL (Ours)":
                cm = rep["confusion_matrix"]
                df_cm = pd.DataFrame(cm, index=PATTERN_CLASSES, columns=PATTERN_CLASSES)
                cm_path = os.path.join(self.results_dir, "confusion_matrix.csv")
                df_cm.to_csv(cm_path)
                self.log.info(f"[Eval] Saved SmellRL confusion matrix to {cm_path}")

        pd.DataFrame(rows).to_csv(os.path.join(self.results_dir, "baseline_comparison.csv"), index=False)
        return results

    def run_per_class_analysis(self) -> dict:
        self.log.info("[Eval] ─── Experiment: Per-class Refactoring F1 Analysis ───")
        true_y = [SMELL_TO_PATTERN_DOMINANT[int(item["y"].item())] for item in self.test_ds]
        
        srl_p1 = self._smellrl_preds(self.test_ds)
        
        rule = RuleBasedAgent()
        rule_p1 = [SMELL_TO_PATTERN_DOMINANT[rule.predict(it)] for it in self.test_ds]

        srl_rep = classification_report_dict(true_y, srl_p1, PATTERN_CLASSES)
        rule_rep = classification_report_dict(true_y, rule_p1, PATTERN_CLASSES)

        rows = []
        for pattern in PATTERN_CLASSES:
            srl_f1 = srl_rep["per_class"].get(pattern, {}).get("f1", 0.0)
            rule_f1 = rule_rep["per_class"].get(pattern, {}).get("f1", 0.0)
            rows.append({
                "Pattern Type": pattern, 
                "SmellRL F1": round(srl_f1, 4), 
                "Rule-Based F1": round(rule_f1, 4),
                "Delta": round(srl_f1 - rule_f1, 4)
            })
            self.log.info(f"[Eval] {pattern:15s} | SmellRL F1={srl_f1:.4f} | Rule-Based F1={rule_f1:.4f} | Δ={srl_f1-rule_f1:+.4f}")

        pd.DataFrame(rows).to_csv(os.path.join(self.results_dir, "exp5_per_class_f1.csv"), index=False)
        return rows

    def run_all(self, training_histories: dict = None) -> dict:
        self.log.info("[Experiments] ═══ Starting Experiments ═══")
        all_results = {}
        
        all_results["baselines"] = self.run_baseline_comparison()
        all_results["per_class"] = self.run_per_class_analysis()
        
        path = os.path.join(self.results_dir, "all_results_summary.json")
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
            
        self.log.info("[Experiments] ═══ All experiments complete ═══")
        return all_results