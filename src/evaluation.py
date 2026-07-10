"""
Evaluation pipeline for Semantic-Aware SmellRL.

Baselines:
  RandomAgent      — uniformly random actions
  RuleBasedAgent   — Designite-like CK metric thresholds
  SVMAgent         — scikit-learn SVC trained on CK metrics

Experiments:
  1. Baseline Comparison (Accuracy, F1, Precision, Recall)
  2. Per-class F1 breakdown (to see if semantics help Feature Envy / God Class)

Note on agent compatibility:
  ExperimentRunner accepts any object that exposes:
    - select_action(item: dict, epsilon: float) -> (int, Tensor)
    - eval() / train()   (standard nn.Module methods)
  Both SupervisedSmellDetector and SmellDetectionAgent satisfy this interface.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple

from src.data import SMELL_CLASSES, SMELL_TO_IDX, SmellDataset, PATTERN_CLASSES

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
    """Uniformly random actions — establishes floor performance."""
    name = "Random Agent"
    def __init__(self, n_smells: int = 5, seed: int = 42):
        self.n_s = n_smells
        np.random.seed(seed)

    def predict(self, _item: dict) -> int:
        return np.random.randint(0, self.n_s)

class RuleBasedAgent:
    """
    Designite-inspired rule-based smell detector.
    """
    name = "Rule-Based (Designite-like)"

    def predict(self, item: dict) -> int:
        x = item["x"]  # [N, 783] raw node feature tensor
        cf = x[0].tolist()  # class node (index 0): 783-dim feature vector

        # Reverse-normalise CK metrics from the class node (indices 3-8)
        wmc = cf[3] * 150.0
        cbo = cf[6] * 40.0
        rfc = cf[7] * 200.0
        loc = cf[8] * 3000.0
        # Correct method/field counts from the node-type one-hot flags:
        # x[:, 1] = is_method, x[:, 2] = is_field across all N nodes
        n_m = max(1, int(x[:, 1].sum().item()))
        n_f = max(0, int(x[:, 2].sum().item()))
        avg_cc = wmc / n_m

        if wmc > 47 or loc > 1000:
            return SMELL_TO_IDX["GodClass"]
        if cbo > 10 and rfc > 50:
            return SMELL_TO_IDX["FeatureEnvy"]
        if (loc / n_m) > 100 or avg_cc > 8:
            return SMELL_TO_IDX["LongMethod"]
        if n_f > 8 and wmc < 10:
            return SMELL_TO_IDX["DataClass"]
        return SMELL_TO_IDX["NoSmell"]

class SVMAgent:
    """SVM + CK metrics baseline trained on the training set."""
    name = "SVM + CK Metrics"

    def __init__(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.clf = SVC(kernel="rbf", C=1.0, probability=False)
        self.scaler = StandardScaler()
        self.trained = False

    def _extract_features(self, item: dict) -> np.ndarray:
        x = item["x"][0].numpy()
        # Extract the core CK metrics + loc: wmc, dit, noc, cbo, rfc, loc
        # We assume they are at indices 3:9 after the 3 one-hot node-type indicators.
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

# ──────────────────────────── Experiment Runner ────────────────────────────

def simulate_refactoring_impact(preds: List[int], test_ds: SmellDataset) -> Tuple[float, float, float]:
    """
    Simulates the impact of recommended refactoring actions on code quality metrics.
    
    If the agent recommends a correct action (reward > 0.0):
      - Facade/Mediator (for GodClass): Reduces class WMC by 40%, CBO by 20%
      - Strategy/ExtractClass (for FeatureEnvy): Reduces class CBO by 30%, RFC by 20%
      - ExtractClass (for LongMethod): Reduces class LOC by 30%, WMC by 15%
      - Observer (for DataClass): Reduces class CBO by 10%
    
    If the agent recommends an incorrect/harmful action:
      - Applying pattern to NoSmell: Increases class CBO by 15% (unnecessary delegation overhead)
      
    Returns:
      wmc_reduction_pct, cbo_reduction_pct, loc_reduction_pct
    """
    initial_wmc = 0.0
    initial_cbo = 0.0
    initial_loc = 0.0
    
    final_wmc = 0.0
    final_cbo = 0.0
    final_loc = 0.0
    
    for item, action in zip(test_ds, preds):
        true_smell = int(item["y"].item())
        x = item["x"][0] # class node
        
        # Reverse-normalize initial metrics
        wmc = float(x[3].item() * 150.0)
        cbo = float(x[6].item() * 40.0)
        loc = float(x[8].item() * 3000.0)
        
        initial_wmc += wmc
        initial_cbo += cbo
        initial_loc += loc
        
        # Apply refactoring impact
        new_wmc = wmc
        new_cbo = cbo
        new_loc = loc
        
        # 0: Strategy, 1: Observer, 2: Facade, 3: Mediator, 4: ExtractClass, 5: None
        if true_smell == 0:  # GodClass
            if action in [2, 3]:  # Facade, Mediator
                new_wmc *= 0.60  # 40% reduction
                new_cbo *= 0.80  # 20% reduction
            elif action == 4:  # ExtractClass
                new_wmc *= 0.80
                new_cbo *= 0.90
        elif true_smell == 1:  # FeatureEnvy
            if action == 0:  # Strategy
                new_cbo *= 0.70  # 30% reduction
                new_wmc *= 0.90
            elif action == 4:  # ExtractClass
                new_cbo *= 0.80
        elif true_smell == 2:  # LongMethod
            if action == 4:  # ExtractClass (representing Extract Method)
                new_loc *= 0.70  # 30% reduction in this class's size / method size
                new_wmc *= 0.85
        elif true_smell == 3:  # DataClass
            if action == 1:  # Observer
                new_cbo *= 0.90
        elif true_smell == 4:  # NoSmell
            if action in [0, 1, 2, 3, 4]:  # False alarm (unnecessary refactoring)
                new_cbo *= 1.15  # 15% increase due to delegation overhead
                
        final_wmc += new_wmc
        final_cbo += new_cbo
        final_loc += new_loc
        
    wmc_red = (initial_wmc - final_wmc) / max(1e-9, initial_wmc) * 100.0
    cbo_red = (initial_cbo - final_cbo) / max(1e-9, initial_cbo) * 100.0
    loc_red = (initial_loc - final_loc) / max(1e-9, initial_loc) * 100.0
    
    return float(wmc_red), float(cbo_red), float(loc_red)

# ──────────────────────────── Experiment Runner ────────────────────────────

class ExperimentRunner:
    """
    Evaluates the agent against baselines and performs F1 breakdown.

    Accepts any `agent` that implements:
        agent.select_action(item: dict, epsilon: float) -> (int, Tensor)
        agent.eval() / agent.train()
    Both SupervisedSmellDetector and SmellDetectionAgent satisfy this contract.
    """
    def __init__(self, agent: nn.Module, train_ds: SmellDataset, test_ds: SmellDataset, cfg: dict, device: torch.device):
        self.agent = agent
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.cfg = cfg
        self.device = device
        self.results_dir = cfg["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
        self.log = logger

    @torch.no_grad()
    def _smellrl_preds(self, ds: SmellDataset) -> List[int]:
        self.agent.eval()
        preds = []
        is_smell_predictor = getattr(self.agent, "n_classes", 5) == 5
        for item in ds:
            action, _ = self.agent.select_action(item, epsilon=0.0)
            if is_smell_predictor:
                action = SMELL_TO_PATTERN_DOMINANT.get(action, 5)
            preds.append(action)
        self.agent.train()
        return preds

    def run_baseline_comparison(self) -> dict:
        self.log.info("[Eval] ─── Experiment: Baseline Comparison ───")
        true_y = [SMELL_TO_PATTERN_DOMINANT[int(item["y"].item())] for item in self.test_ds]
        methods = {}

        rand = RandomAgent()
        methods["Random"] = [SMELL_TO_PATTERN_DOMINANT[rand.predict(it)] for it in self.test_ds]

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
            wmc_red, cbo_red, loc_red = simulate_refactoring_impact(preds, self.test_ds)
            results[name] = rep
            results[name]["wmc_reduction"] = wmc_red
            results[name]["cbo_reduction"] = cbo_red
            results[name]["loc_reduction"] = loc_red
            
            rows.append({
                "Method": name, 
                "Accuracy": round(rep["accuracy"], 4), 
                "F1": round(rep["f1"], 4), 
                "Precision": round(rep["precision"], 4), 
                "Recall": round(rep["recall"], 4),
                "WMC_Complexity_Reduction(%)": round(wmc_red, 2),
                "CBO_Coupling_Reduction(%)": round(cbo_red, 2),
                "LOC_Size_Reduction(%)": round(loc_red, 2)
            })
            self.log.info(f"[Eval] {name:15s} | Acc={rep['accuracy']:.4f} | F1={rep['f1']:.4f} | WMC_red={wmc_red:.1f}% | CBO_red={cbo_red:.1f}%")
            
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