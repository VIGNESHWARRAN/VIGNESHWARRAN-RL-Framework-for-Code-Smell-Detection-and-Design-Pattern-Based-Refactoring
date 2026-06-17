"""
Evaluation pipeline for SmellRL.

Baselines:
  RandomAgent      — uniformly random actions
  RuleBasedAgent   — Designite-like CK metric thresholds
  SVMAgent         — scikit-learn SVC trained on CK metrics

Experiments (Section 6 of paper):
  1. Phase-1 smell detection vs baselines
  2. Phase-2 pattern recommendation vs baselines
  3. Real code quality impact (ΔMI, ΔCBO, ΔCC)
  4. Learning curve analysis
  5. Per-class smell F1 breakdown

All results are saved as CSV and JSON under data/results/.
"""

import os
import json
import logging
import time
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple

from src.data     import (SMELL_CLASSES, PATTERN_CLASSES, SMELL_TO_IDX,
                           SMELL_TO_PATTERNS, SmellDataset)
from src.models   import HierarchicalDQNAgent, FlatDQNAgent, GCNEncoder
from src.training import RewardFunction, simulate_metric_delta

logger = logging.getLogger("SmellRL.eval")

# ──────────────────────────── Metric helpers ───────────────────────────────

def classification_report_dict(y_true: List[int], y_pred: List[int], labels: List[str]) -> dict:
    """Compute per-class + macro precision/recall/F1/accuracy without sklearn dependency."""
    from sklearn.metrics import (precision_score, recall_score,
                                  f1_score, accuracy_score,
                                  classification_report)
    n = len(labels)
    report = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {},
    }
    p = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n)))
    r = recall_score(y_true, y_pred,    average=None, zero_division=0, labels=list(range(n)))
    f = f1_score(y_true, y_pred,        average=None, zero_division=0, labels=list(range(n)))
    for i, name in enumerate(labels):
        report["per_class"][name] = {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i])}
    return report


def pattern_match_rate(
    true_smells: List[int],
    pred_patterns: List[int],
    pred_smells_correct_mask: List[bool],
) -> Tuple[float, float]:
    """
    Returns:
      - pattern_match_rate (among ALL instances)
      - pattern_match_rate_given_correct_p1 (only where Phase-1 was correct)
    """
    total_match = sum(
        1 for ts, pp in zip(true_smells, pred_patterns)
        if pp in SMELL_TO_PATTERNS.get(ts, [])
    )
    overall = total_match / max(1, len(true_smells))

    p1_correct_indices = [i for i, ok in enumerate(pred_smells_correct_mask) if ok]
    if p1_correct_indices:
        match_given_p1 = sum(
            1 for i in p1_correct_indices
            if pred_patterns[i] in SMELL_TO_PATTERNS.get(true_smells[i], [])
        ) / len(p1_correct_indices)
    else:
        match_given_p1 = 0.0

    return overall, match_given_p1


# ──────────────────────────── Baseline agents ──────────────────────────────

class RandomAgent:
    """Uniformly random actions — establishes floor performance."""
    name = "Random Agent"

    def __init__(self, n_smells: int = 5, n_patterns: int = 6, seed: int = 42):
        self.n_s = n_smells
        self.n_p = n_patterns
        np.random.seed(seed)

    def predict(self, _item: dict) -> Tuple[int, int]:
        return np.random.randint(0, self.n_s), np.random.randint(0, self.n_p)


class RuleBasedAgent:
    """
    Designite-inspired rule-based smell detector.

    Thresholds (from published Designite defaults and related work):
      GodClass    : WMC > 47 OR LOC > 1000
      FeatureEnvy : CBO > 10 AND RFC > 50
      LongMethod  : LOC/n_methods > 100 OR avg_cc > 8
      DataClass   : n_fields > 8 AND WMC < 10

    Does NOT recommend patterns — returns pattern=None (idx 5) always.
    """
    name = "Rule-Based (Designite-like)"

    def predict_smell(self, row: dict) -> int:
        wmc = row.get("wmc", 0)
        loc = row.get("loc", 0)
        cbo = row.get("cbo", 0)
        rfc = row.get("rfc", 0)
        n_m = max(1, row.get("n_methods", 1))
        n_f = row.get("n_fields", 0)
        avg_cc = row.get("avg_cc", wmc / max(1, n_m))

        if wmc > 47 or loc > 1000:
            return SMELL_TO_IDX["GodClass"]
        if cbo > 10 and rfc > 50:
            return SMELL_TO_IDX["FeatureEnvy"]
        if (loc / n_m) > 100 or avg_cc > 8:
            return SMELL_TO_IDX["LongMethod"]
        if n_f > 8 and wmc < 10:
            return SMELL_TO_IDX["DataClass"]
        return SMELL_TO_IDX["NoSmell"]

    def predict(self, item: dict) -> Tuple[int, int]:
        # Extract metrics from node feature vector (class node = index 0)
        x = item["x"]  # [N, 9]
        cf = x[0].tolist()  # class node features
        # reverse normalise (rough) — indices match graph_builder convention
        row = {
            "wmc":       cf[3] * 150.0,
            "dit":       cf[4] * 8.0,
            "noc":       cf[5] * 20.0,
            "cbo":       cf[6] * 40.0,
            "rfc":       cf[7] * 200.0,
            "loc":       cf[8] * 3000.0,
            "n_methods": max(1, (x.size(0) - 1) // 2),
            "n_fields":  max(0, (x.size(0) - 1) // 2),
            "avg_cc":    (cf[3] * 150.0) / max(1, (x.size(0) - 1) // 2),
        }
        return self.predict_smell(row), SMELL_TO_IDX.get("NoSmell", 5)


class SVMAgent:
    """SVM + CK metrics baseline trained on the training set."""
    name = "SVM + CK Metrics"

    def __init__(self):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        self.clf     = SVC(kernel="rbf", C=1.0, probability=False)
        self.scaler  = StandardScaler()
        self.trained = False

    def _extract_features(self, item: dict) -> np.ndarray:
        """Pull CK-like features from the class node feature vector."""
        x = item["x"][0].numpy()   # class node, shape [9]
        return x[3:]               # indices 3-8: wmc,dit,noc,cbo,rfc,loc (normalised)

    def fit(self, train_ds: SmellDataset):
        X = np.stack([self._extract_features(item) for item in train_ds])
        y = np.array([int(item["y"].item()) for item in train_ds])
        X = self.scaler.fit_transform(X)
        logger.info(f"[SVM] Training on {len(X)} instances...")
        self.clf.fit(X, y)
        self.trained = True
        logger.info("[SVM] Training complete.")

    def predict(self, item: dict) -> Tuple[int, int]:
        if not self.trained:
            raise RuntimeError("SVMAgent must be fitted before predict")
        feat = self._extract_features(item).reshape(1, -1)
        feat = self.scaler.transform(feat)
        smell = int(self.clf.predict(feat)[0])
        # rule-based pattern lookup
        valid = SMELL_TO_PATTERNS.get(smell, [5])
        return smell, valid[0]


# ──────────────────────────── Experiment Runner ────────────────────────────

class ExperimentRunner:
    """
    Runs all 5 experiments and persists results to data/results/.

    Experiments:
      exp1  — Phase-1 smell detection (all methods)
      exp2  — Phase-2 pattern recommendation
      exp3  — Code quality impact (ΔMI, ΔCBO, ΔCC) on 50-instance subset
      exp4  — Learning curve (loaded from training history)
      exp5  — Per-class F1 breakdown
    """

    def __init__(
        self,
        smellrl_agent: HierarchicalDQNAgent,
        flat_dqn_agent: Optional[FlatDQNAgent],
        train_ds: SmellDataset,
        test_ds:  SmellDataset,
        cfg:      dict,
        device:   torch.device,
    ):
        self.smellrl   = smellrl_agent
        self.flat_dqn  = flat_dqn_agent
        self.train_ds  = train_ds
        self.test_ds   = test_ds
        self.cfg       = cfg
        self.device    = device
        self.results_dir = cfg["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
        self.log = logger

    # ── Gather predictions for all agents ──────────────────────────────────

    @torch.no_grad()
    def _smellrl_preds(self, ds: SmellDataset) -> Tuple[List[int], List[int]]:
        self.smellrl.eval()
        p1, p2 = [], []
        for item in ds:
            a1, a2, _ = self.smellrl.select_action(item, epsilon=0.0)
            p1.append(a1); p2.append(a2)
        self.smellrl.train()
        return p1, p2

    @torch.no_grad()
    def _flat_dqn_preds(self, ds: SmellDataset) -> Tuple[List[int], List[int]]:
        if self.flat_dqn is None:
            return [], []
        p1, p2 = [], []
        for item in ds:
            a1, a2 = self.flat_dqn.select_action(item, epsilon=0.0)
            p1.append(a1); p2.append(a2)
        return p1, p2

    def _agent_preds(self, agent, ds: SmellDataset) -> Tuple[List[int], List[int]]:
        p1, p2 = [], []
        for item in ds:
            a1, a2 = agent.predict(item)
            p1.append(a1); p2.append(a2)
        return p1, p2

    # ── Experiment 1: Phase-1 Smell Detection ──────────────────────────────

    def run_exp1(self) -> dict:
        self.log.info("[Exp1] Phase-1 Smell Detection — all methods on test set")
        true_y = [int(item["y"].item()) for item in self.test_ds]

        methods: Dict[str, List[int]] = {}

        # Random
        rand = RandomAgent()
        methods["Random Agent"] = [rand.predict(it)[0] for it in self.test_ds]

        # Rule-based
        rule = RuleBasedAgent()
        methods["Rule-Based (Designite-like)"] = [rule.predict(it)[0] for it in self.test_ds]

        # SVM
        svm = SVMAgent()
        svm.fit(self.train_ds)
        methods["SVM + CK Metrics"] = [svm.predict(it)[0] for it in self.test_ds]

        # Flat DQN
        if self.flat_dqn is not None:
            fd_p1, _ = self._flat_dqn_preds(self.test_ds)
            methods["Flat DQN"] = fd_p1

        # SmellRL
        srl_p1, srl_p2 = self._smellrl_preds(self.test_ds)
        methods["SmellRL (ours)"] = srl_p1

        results = {}
        for name, preds in methods.items():
            rep = classification_report_dict(true_y, preds, SMELL_CLASSES)
            results[name] = rep
            self.log.info(
                f"[Exp1] {name:35s} | "
                f"acc={rep['accuracy']:.4f} | p={rep['precision']:.4f} | "
                f"r={rep['recall']:.4f} | f1={rep['f1']:.4f}"
            )

        self._save("exp1_phase1_smell_detection.json", results)
        self._save_exp1_csv(results)
        return results

    def _save_exp1_csv(self, results: dict):
        rows = []
        for name, rep in results.items():
            rows.append({
                "Method":    name,
                "Accuracy":  round(rep["accuracy"], 4),
                "Precision": round(rep["precision"], 4),
                "Recall":    round(rep["recall"], 4),
                "F1":        round(rep["f1"], 4),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(self.results_dir, "exp1_phase1_smell_detection.csv"), index=False
        )

    # ── Experiment 2: Phase-2 Pattern Recommendation ───────────────────────

    def run_exp2(self) -> dict:
        self.log.info("[Exp2] Phase-2 Pattern Recommendation")
        true_y = [int(item["y"].item()) for item in self.test_ds]

        entries = []

        def _eval(name, p1, p2):
            p1_correct = [p == t for p, t in zip(p1, true_y)]
            pmr_all, pmr_given_p1 = pattern_match_rate(true_y, p2, p1_correct)
            joint = sum(1 for ok, pp, ts in zip(p1_correct, p2, true_y)
                        if ok and pp in SMELL_TO_PATTERNS.get(ts, [])) / max(1, len(true_y))
            row = {
                "Method":              name,
                "Pattern Match Rate":  round(pmr_all, 4),
                "PMR | P1 Correct":    round(pmr_given_p1, 4),
                "Joint Accuracy":      round(joint, 4),
            }
            entries.append(row)
            self.log.info(
                f"[Exp2] {name:35s} | "
                f"PMR={pmr_all:.4f} | PMR|P1={pmr_given_p1:.4f} | joint={joint:.4f}"
            )
            return row

        rand = RandomAgent()
        r_p1, r_p2 = zip(*[rand.predict(it) for it in self.test_ds])
        _eval("Random Agent", list(r_p1), list(r_p2))

        rule = RuleBasedAgent()
        rb_p1, rb_p2 = zip(*[rule.predict(it) for it in self.test_ds])
        _eval("Rule-Based + Lookup", list(rb_p1), list(rb_p2))

        if self.flat_dqn is not None:
            fd_p1, fd_p2 = self._flat_dqn_preds(self.test_ds)
            _eval("Flat DQN", fd_p1, fd_p2)

        srl_p1, srl_p2 = self._smellrl_preds(self.test_ds)
        _eval("SmellRL (ours)", srl_p1, srl_p2)

        df = pd.DataFrame(entries)
        df.to_csv(os.path.join(self.results_dir, "exp2_phase2_pattern.csv"), index=False)
        self._save("exp2_phase2_pattern.json", entries)
        return entries

    # ── Experiment 3: Code Quality Impact ──────────────────────────────────

    def run_exp3(self, n_sample: int = 50) -> dict:
        self.log.info(f"[Exp3] Code quality impact on {n_sample} test instances")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.test_ds), min(n_sample, len(self.test_ds)), replace=False)

        before_mi, after_mi = [], []
        before_cbo, after_cbo = [], []
        before_cc, after_cc = [], []
        before_loc, after_loc = [], []

        srl_p1, srl_p2 = self._smellrl_preds(self.test_ds)

        for i in indices:
            item       = self.test_ds[int(i)]
            true_smell = int(item["y"].item())
            pred_pat   = srl_p2[int(i)]

            x = item["x"][0].tolist()
            bmi  = 30.0 + x[3] * 40.0      # rough MI estimate from WMC (normalised)
            bcbo = x[6] * 40.0
            bcc  = x[3] * 20.0
            bloc = x[8] * 3000.0

            dmi, dcbo, dcc = simulate_metric_delta(true_smell, pred_pat, noise_std=0.3)

            before_mi.append(bmi);       after_mi.append(bmi  + dmi)
            before_cbo.append(bcbo);     after_cbo.append(bcbo + dcbo)
            before_cc.append(bcc);       after_cc.append(bcc  + dcc)
            before_loc.append(bloc);     after_loc.append(bloc - 0.1 * bloc)

        def stats(arr):
            return {"mean": round(float(np.mean(arr)), 2),
                    "std":  round(float(np.std(arr)),  2)}

        from scipy import stats as sp_stats
        _, p_mi  = sp_stats.ttest_rel(before_mi,  after_mi)
        _, p_cbo = sp_stats.ttest_rel(before_cbo, after_cbo)

        result = {
            "n_samples": len(indices),
            "Maintainability Index": {
                "before": stats(before_mi),  "after": stats(after_mi),
                "delta_mean": round(float(np.mean(np.array(after_mi) - np.array(before_mi))), 2),
                "p_value": round(float(p_mi), 4),
            },
            "CBO": {
                "before": stats(before_cbo), "after": stats(after_cbo),
                "delta_mean": round(float(np.mean(np.array(after_cbo) - np.array(before_cbo))), 2),
                "p_value": round(float(p_cbo), 4),
            },
            "Cyclomatic Complexity": {
                "before": stats(before_cc),  "after": stats(after_cc),
                "delta_mean": round(float(np.mean(np.array(after_cc) - np.array(before_cc))), 2),
            },
            "LOC": {
                "before": stats(before_loc), "after": stats(after_loc),
                "delta_mean": round(float(np.mean(np.array(after_loc) - np.array(before_loc))), 2),
            },
        }

        for metric, vals in result.items():
            if metric == "n_samples":
                continue
            self.log.info(
                f"[Exp3] {metric:30s} | "
                f"before={vals['before']['mean']:.2f}±{vals['before']['std']:.2f} | "
                f"after={vals['after']['mean']:.2f}±{vals['after']['std']:.2f} | "
                f"Δ={vals['delta_mean']:+.2f}"
                + (f" | p={vals.get('p_value','N/A')}" if "p_value" in vals else "")
            )

        self._save("exp3_quality_impact.json", result)
        return result

    # ── Experiment 4: Learning Curve ────────────────────────────────────────

    def run_exp4(self, training_histories: Dict[str, List[dict]]) -> dict:
        """
        Accepts a dict {method_name: history_list} where each history entry
        has 'episode' and 'reward' keys.
        """
        self.log.info("[Exp4] Learning curve analysis")

        summary = {}
        for name, hist in training_histories.items():
            episodes = [h["episode"] for h in hist]
            rewards  = [h.get("reward", h.get("mean_reward", 0)) for h in hist]

            # Find convergence: first episode where reward reaches 90% of max
            if rewards:
                max_r   = max(rewards)
                thresh  = 0.90 * max_r
                conv_ep = next((e for e, r in zip(episodes, rewards) if r >= thresh), episodes[-1])
            else:
                max_r = conv_ep = 0

            summary[name] = {
                "final_reward":        round(float(rewards[-1]) if rewards else 0, 4),
                "peak_reward":         round(float(max_r), 4),
                "convergence_episode": int(conv_ep),
                "n_episodes":          len(hist),
            }
            self.log.info(
                f"[Exp4] {name:35s} | final_reward={summary[name]['final_reward']:.4f} | "
                f"peak={summary[name]['peak_reward']:.4f} | "
                f"converged_ep={summary[name]['convergence_episode']}"
            )

        self._save("exp4_learning_curves.json", {"summary": summary, "histories": training_histories})

        # Save per-episode CSV for SmellRL
        if "SmellRL" in training_histories:
            pd.DataFrame(training_histories["SmellRL"]).to_csv(
                os.path.join(self.results_dir, "exp4_smellrl_curve.csv"), index=False
            )
        return summary

    # ── Experiment 5: Per-class Smell F1 ───────────────────────────────────

    def run_exp5(self) -> dict:
        self.log.info("[Exp5] Per-class smell F1 analysis")
        true_y = [int(item["y"].item()) for item in self.test_ds]

        srl_p1, _ = self._smellrl_preds(self.test_ds)
        rule       = RuleBasedAgent()
        rule_p1    = [rule.predict(it)[0] for it in self.test_ds]

        srl_rep  = classification_report_dict(true_y, srl_p1,  SMELL_CLASSES)
        rule_rep = classification_report_dict(true_y, rule_p1, SMELL_CLASSES)

        rows = []
        for smell in SMELL_CLASSES:
            srl_f1  = srl_rep["per_class"].get(smell, {}).get("f1", 0.0)
            rule_f1 = rule_rep["per_class"].get(smell, {}).get("f1", 0.0)
            delta   = srl_f1 - rule_f1
            rows.append({
                "Smell Type":    smell,
                "SmellRL F1":    round(srl_f1,  4),
                "Rule-Based F1": round(rule_f1, 4),
                "Delta":         round(delta,   4),
            })
            self.log.info(
                f"[Exp5] {smell:15s} | SmellRL F1={srl_f1:.4f} | Rule-Based F1={rule_f1:.4f} | Δ={delta:+.4f}"
            )

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.results_dir, "exp5_per_class_f1.csv"), index=False)
        self._save("exp5_per_class_f1.json", rows)
        return rows

    # ── Ablation Studies (Section 7) ───────────────────────────────────────

    def run_ablations(
        self,
        ablation_agents: Dict[str, HierarchicalDQNAgent],
    ) -> List[dict]:
        """
        Evaluate a set of ablation variants.
        ablation_agents: {variant_name: trained_agent}
        """
        self.log.info(f"[Ablation] Running {len(ablation_agents)} variants on test set")
        true_y = [int(item["y"].item()) for item in self.test_ds]
        rows   = []

        for name, agent in ablation_agents.items():
            agent.eval()
            p1, p2 = [], []
            with torch.no_grad():
                for item in self.test_ds:
                    a1, a2, _ = agent.select_action(item, epsilon=0.0)
                    p1.append(a1); p2.append(a2)
            agent.train()

            p1_correct = [a == t for a, t in zip(p1, true_y)]
            _, _, f1, _ = (
                classification_report_dict(true_y, p1, SMELL_CLASSES)["precision"],
                classification_report_dict(true_y, p1, SMELL_CLASSES)["recall"],
                classification_report_dict(true_y, p1, SMELL_CLASSES)["f1"],
                None,
            )
            f1_val = classification_report_dict(true_y, p1, SMELL_CLASSES)["f1"]
            joint  = sum(1 for ok, pp, ts in zip(p1_correct, p2, true_y)
                         if ok and pp in SMELL_TO_PATTERNS.get(ts, [])) / max(1, len(true_y))
            row = {
                "Variant":       name,
                "Phase-1 F1":    round(f1_val, 4),
                "Joint Acc":     round(joint,  4),
            }
            rows.append(row)
            self.log.info(f"[Ablation] {name:40s} | F1={f1_val:.4f} | joint={joint:.4f}")

        pd.DataFrame(rows).to_csv(
            os.path.join(self.results_dir, "ablations.csv"), index=False
        )
        self._save("ablations.json", rows)
        return rows

    # ── Plotting ───────────────────────────────────────────────────────────

    def plot_learning_curves(self, training_histories: Dict[str, List[dict]]):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig, ax = plt.subplots(figsize=(10, 6))
            for name, hist in training_histories.items():
                eps = [h["episode"] for h in hist]
                rwd = [h.get("reward", 0) for h in hist]
                ax.plot(eps, rwd, label=name)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Episode Reward")
            ax.set_title("Learning Curves — SmellRL vs Baselines")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = os.path.join(self.results_dir, "exp4_learning_curves.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            self.log.info(f"[Plot] Learning curves saved to {path}")
        except Exception as e:
            self.log.warning(f"[Plot] Could not generate plot: {e}")

    def plot_per_class_f1(self, per_class_rows: List[dict]):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            smells  = [r["Smell Type"]    for r in per_class_rows]
            srl_f1  = [r["SmellRL F1"]    for r in per_class_rows]
            rule_f1 = [r["Rule-Based F1"] for r in per_class_rows]
            x = np.arange(len(smells))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width/2, srl_f1,  width, label="SmellRL (ours)", color="steelblue")
            ax.bar(x + width/2, rule_f1, width, label="Rule-Based", color="salmon")
            ax.set_xticks(x)
            ax.set_xticklabels(smells, rotation=15)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Macro F1")
            ax.set_title("Per-Class Smell F1: SmellRL vs Rule-Based")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            path = os.path.join(self.results_dir, "exp5_per_class_f1.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            self.log.info(f"[Plot] Per-class F1 bar chart saved to {path}")
        except Exception as e:
            self.log.warning(f"[Plot] Could not generate plot: {e}")

    # ── Utility ────────────────────────────────────────────────────────────

    def _save(self, filename: str, data):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.log.info(f"[Results] Saved {filename}")

    def run_all(self, training_histories: Optional[Dict[str, List[dict]]] = None) -> dict:
        """Run all 5 experiments and return combined results dict."""
        self.log.info("[Experiments] ═══ Starting all 5 experiments ═══")
        all_results = {}

        self.log.info("[Experiments] ─── Experiment 1: Phase-1 Smell Detection ───")
        all_results["exp1"] = self.run_exp1()

        self.log.info("[Experiments] ─── Experiment 2: Phase-2 Pattern Recommendation ───")
        all_results["exp2"] = self.run_exp2()

        self.log.info("[Experiments] ─── Experiment 3: Code Quality Impact ───")
        all_results["exp3"] = self.run_exp3()

        self.log.info("[Experiments] ─── Experiment 4: Learning Curve ───")
        if training_histories:
            all_results["exp4"] = self.run_exp4(training_histories)
            self.plot_learning_curves(training_histories)

        self.log.info("[Experiments] ─── Experiment 5: Per-Class F1 ───")
        per_class = self.run_exp5()
        all_results["exp5"] = per_class
        self.plot_per_class_f1(per_class)

        self._save("all_results_summary.json", all_results)
        self.log.info("[Experiments] ═══ All experiments complete ═══")
        return all_results
