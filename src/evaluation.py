"""
Evaluation pipeline for Semantic-Aware SmellRL.

Baselines:
  RandomAgent      — uniformly random actions
  RuleBasedAgent   — Designite-like CK metric thresholds
  SVMAgent         — scikit-learn SVC trained on CK metrics

Experiments:
  1. Baseline Comparison (Accuracy, F1, Precision, Recall)
  2. Per-class F1 breakdown (to see if semantics help Feature Envy / God Class)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

from src.data import SMELL_CLASSES, SMELL_TO_IDX, SmellDataset
from src.models import SmellDetectionAgent

logger = logging.getLogger("SmellRL.eval")

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
        x = item["x"]  # [N, Node_Feature_Dim]
        cf = x[0].tolist()  # class node features
        
        # reverse normalise (rough approximation)
        wmc = cf[3] * 150.0
        cbo = cf[6] * 40.0
        rfc = cf[7] * 200.0
        loc = cf[8] * 3000.0 if len(cf) > 8 else 0  # Support dropping LOC
        n_m = max(1, (x.size(0) - 1) // 2)
        n_f = max(0, (x.size(0) - 1) // 2)
        avg_cc = wmc / max(1, n_m)

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

class ExperimentRunner:
    """
    Evaluates the agent against baselines and performs F1 breakdown.
    """
    def __init__(self, agent: SmellDetectionAgent, train_ds: SmellDataset, test_ds: SmellDataset, cfg: dict, device: torch.device):
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
        for item in ds:
            action, _ = self.agent.select_action(item, epsilon=0.0)
            preds.append(action)
        self.agent.train()
        return preds

    def run_baseline_comparison(self) -> dict:
        self.log.info("[Eval] ─── Experiment: Baseline Comparison ───")
        true_y = [int(item["y"].item()) for item in self.test_ds]
        methods = {}

        rand = RandomAgent()
        methods["Random"] = [rand.predict(it) for it in self.test_ds]

        rule = RuleBasedAgent()
        methods["Rule-Based"] = [rule.predict(it) for it in self.test_ds]

        svm = SVMAgent()
        svm.fit(self.train_ds)
        methods["SVM"] = [svm.predict(it) for it in self.test_ds]

        methods["SmellRL (Ours)"] = self._smellrl_preds(self.test_ds)

        results = {}
        rows = []
        for name, preds in methods.items():
            rep = classification_report_dict(true_y, preds, SMELL_CLASSES)
            results[name] = rep
            rows.append({
                "Method": name, 
                "Accuracy": round(rep["accuracy"], 4), 
                "F1": round(rep["f1"], 4), 
                "Precision": round(rep["precision"], 4), 
                "Recall": round(rep["recall"], 4)
            })
            self.log.info(f"[Eval] {name:15s} | Acc={rep['accuracy']:.4f} | F1={rep['f1']:.4f}")
            
            if name == "SmellRL (Ours)":
                cm = rep["confusion_matrix"]
                df_cm = pd.DataFrame(cm, index=SMELL_CLASSES, columns=SMELL_CLASSES)
                cm_path = os.path.join(self.results_dir, "confusion_matrix.csv")
                df_cm.to_csv(cm_path)
                self.log.info(f"[Eval] Saved SmellRL confusion matrix to {cm_path}")

        pd.DataFrame(rows).to_csv(os.path.join(self.results_dir, "baseline_comparison.csv"), index=False)
        return results

    def run_per_class_analysis(self) -> dict:
        self.log.info("[Eval] ─── Experiment: Per-class Smell F1 Analysis ───")
        true_y = [int(item["y"].item()) for item in self.test_ds]
        
        srl_p1 = self._smellrl_preds(self.test_ds)
        
        rule = RuleBasedAgent()
        rule_p1 = [rule.predict(it) for it in self.test_ds]

        srl_rep = classification_report_dict(true_y, srl_p1, SMELL_CLASSES)
        rule_rep = classification_report_dict(true_y, rule_p1, SMELL_CLASSES)

        rows = []
        for smell in SMELL_CLASSES:
            srl_f1 = srl_rep["per_class"].get(smell, {}).get("f1", 0.0)
            rule_f1 = rule_rep["per_class"].get(smell, {}).get("f1", 0.0)
            rows.append({
                "Smell Type": smell, 
                "SmellRL F1": round(srl_f1, 4), 
                "Rule-Based F1": round(rule_f1, 4),
                "Delta": round(srl_f1 - rule_f1, 4)
            })
            self.log.info(f"[Eval] {smell:15s} | SmellRL F1={srl_f1:.4f} | Rule-Based F1={rule_f1:.4f} | Δ={srl_f1-rule_f1:+.4f}")

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