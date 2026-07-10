# Semantic-Aware SmellRL v3 — Complete Technical Reference

**SmellRL v3** is a reinforcement learning system that combines **Relational Graph Attention Networks (R-GAT)**, **GraphCodeBERT semantic embeddings**, and a **genuine 2-step Deep Q-Network (γ=0.9)** to recommend software refactoring design patterns on Java source code from the **SmellyCode++** benchmark.

This document is the single source of truth for all architectural decisions, dataset choices, training procedures, reward design, ablation methodology, and hyperparameters — with explicit justifications for every decision.

**Dataset:** SmellyCode++ (Nature Scientific Data, 2025) — 107,554 Java samples, 103 open-source projects  
**Primary Claim:** Semantic embeddings (GraphCodeBERT) + structural graph reasoning (R-GAT) + 2-step RL outperforms structure-only and non-RL baselines on minority-class Macro F1.

---

## Table of Contents

1. [Problem Statement & Contributions](#1-problem-statement--contributions)
2. [Dataset: SmellyCode++](#2-dataset-smellycode)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 0 — Data Loading & Preprocessing](#4-stage-0--data-loading--preprocessing)
5. [Stage 1 — AST Graph Construction](#5-stage-1--ast-graph-construction)
6. [Stage 2 — NodeFeatureFusion (18.75% Semantic Bottleneck)](#6-stage-2--nodefeaturefusion-1875-semantic-bottleneck)
7. [Stage 3 — Semantic Embeddings: GraphCodeBERT](#7-stage-3--semantic-embeddings-graphcodebert)
8. [Stage 4 — R-GAT Encoder Architecture](#8-stage-4--r-gat-encoder-architecture)
9. [Stage 5 — RL Environment & State Transitions](#9-stage-5--rl-environment--state-transitions)
10. [Stage 6 — DQN Agent Architecture](#10-stage-6--dqn-agent-architecture)
11. [Stage 7 — 2-Step RL Training Loop](#11-stage-7--2-step-rl-training-loop)
12. [Stage 8 — Evaluation & Baselines](#12-stage-8--evaluation--baselines)
13. [Ablation Studies (Core Experimental Design)](#13-ablation-studies-core-experimental-design)
14. [Complete Hyperparameter Table](#14-complete-hyperparameter-table)
15. [Design Principles](#15-design-principles)
16. [Repository Structure](#16-repository-structure)
17. [Dependencies](#17-dependencies)
18. [Reproducibility & Execution Order](#18-reproducibility--execution-order)

---

## 1. Problem Statement & Contributions

### Problem

Traditional code smell detectors (Designite, PMD) apply hand-coded metric thresholds. They lack semantic understanding of code intent, produce binary smell/no-smell outputs, and do not recommend what to do about a smell. Machine learning classifiers improve detection but still cannot model the fact that multiple refactoring patterns may be valid for a given smell, with varying degrees of optimality.

### What SmellRL v3 Does

- Formulates refactoring recommendation as a **genuine 2-step Markov Decision Process** (γ=0.9), where the agent observes a smelly code graph, selects a refactoring pattern, and then observes the post-refactoring code state to make a refinement decision.
- Fuses **Halstead/complexity structural metrics** with **GraphCodeBERT semantic embeddings** using a controlled 18.75% semantic bottleneck that prevents high-dimensional embeddings from drowning structural signals.
- Uses **SmellyCode++** (107,554 samples, Nature Scientific Data 2025) — a significantly larger and cleaner dataset than prior work on MLCQ.

### Refactoring Action Space

| Index | Pattern | Primary Target | Citation |
|---|---|---|---|
| 0 | `Strategy` | FeatureEnvy | Fowler (1999) §7 |
| 1 | `Observer` | DataClass | Fowler (1999) §11 |
| 2 | `Facade` | GodClass | Fowler (1999) §12 |
| 3 | `Mediator` | GodClass (coupling) | Brown et al. (1998) §3 |
| 4 | `ExtractClass` | LongMethod / GodClass | Fowler (1999) §6 |
| 5 | `NoRefactor` | NoSmell | — |

### Contributions

1. **Genuine 2-step RL MDP (γ=0.9):** Unlike γ=0 contextual bandits (equivalent to weighted classifiers), the agent learns temporal credit assignment across a simulated post-refactoring state transition.
2. **Halstead + GraphCodeBERT fusion with controlled bottleneck:** 18.75% semantic cap prevents dimensionality flooding while preserving semantic discriminability.
3. **SmellyCode++ at scale:** First application of the 107K-sample SmellyCode++ benchmark to RL-based refactoring recommendation; enables statistically significant multi-seed evaluation.
4. **Semantic ablation as core claim:** Controlled ablation (`Full` vs `NoSemantic` vs `MetricsOnly`) using identical RL training code with zeroed semantic dims — making the semantic contribution empirically verifiable.

---

## 2. Dataset: SmellyCode++

**DOI:** `10.6084/m9.figshare.28519385.v1`  
**Published:** Nature Scientific Data, July 2025  
**Size:** 107,554 Java samples from 103 open-source projects  

### Why SmellyCode++ Over MLCQ

| Property | MLCQ | SmellyCode++ |
|---|---|---|
| Size | ~11K usable rows | 107,554 rows |
| Sources | Mixed, single study | 103 open-source Java projects |
| Label quality | Low inter-rater κ for minor severity | Nature peer-reviewed methodology |
| Metrics | CK (WMC, DIT, CBO, RFC, LOC) | 14 Halstead + complexity metrics |
| Multi-label | No | Yes (4 binary columns) |
| Code included | Via separate download | Inline `Code` column |

### Schema

| Column | Type | Description |
|---|---|---|
| `Code` | string | Preprocessed Java source (comments stripped) |
| `GodClass` | 0/1 | Binary smell label |
| `FeatureEnvy` | 0/1 | Binary smell label |
| `LongMethod` | 0/1 | Binary smell label |
| `DataClass` | 0/1 | Binary smell label |
| `lloc` | float | Logical lines of code |
| `cyclomatic` | float | Cyclomatic complexity |
| `n1` | float | Distinct operators (Halstead η₁) |
| `n2` | float | Distinct operands (Halstead η₂) |
| `N1` | float | Total operators |
| `N2` | float | Total operands |
| `N` | float | Halstead program length |
| `Nhat` | float | Estimated program length |
| `V` | float | Halstead Volume |
| `D` | float | Halstead Difficulty |
| `E` | float | Halstead Effort |
| `T` | float | Halstead Time |
| `B` | float | Halstead Bugs estimate |

### Multi-Label to Single Dominant Label

SmellyCode++ is multi-label (a class can have GodClass=1 AND DataClass=1). We convert to a single dominant label using priority order:

```
GodClass > FeatureEnvy > LongMethod > DataClass > NoSmell
```

The full multi-label binary vector is stored as `co_smell_mask` per graph item for reward bonus computation (co-occurring smells receive partial positive reward for adjacent patterns).

### Class Distribution & Balancing

`NoSmell` instances are downsampled to `3.0 × max_smell_class_count` to prevent majority-class bias. Distribution is logged before and after downsampling at INFO level.

### Train / Validation / Test Split

Stratified by `smell_idx` using `sklearn.model_selection.train_test_split`:

| Split | Ratio |
|---|---|
| Train | 70% |
| Val | 15% — used for Phase 1 winner selection (NOT test) |
| Test | 15% — held-out, used only for final evaluation |

---

## 3. Pipeline Overview

```
SmellyCode++ CSV (107,554 rows)
        │
        ▼
[Stage 0] load_smellycode()
          Multi-label → dominant smell · Halstead metrics parsed
          NoSmell downsampled (3× max smell) · Logged fully
        │
        ▼
[Stage 1] AST Graph Construction (_build_ast_graph)
          javalang parses Code column · Class/method/field nodes
          CONTAINS + CALLS + ACCESSES_FIELD edges · GraphCodeBERT cached
        │
        ▼
[Stage 2] NodeFeatureFusion
          15-dim structural [3 node-type + 6 Halstead + 6 data-usage]
          768-dim semantic [GraphCodeBERT CLS] → fused 128-dim
          Bottleneck: 104 struct : 24 semantic (18.75% semantic)
        │
        ▼
[Stage 3] GCN Pre-training (warm-start)
          RGATEncoder + linear head · Cross-entropy · 100 epochs
          Head discarded · Encoder weights loaded into DQN agent
        │
        ▼
[Stage 4] 2-Step RL Training (PRIMARY — DQN, γ=0.9)
          Step 1: Agent observes code graph → picks refactoring pattern
          Transition: apply_simulated_refactoring() → new metric state
          Step 2: Agent observes new state → refinement decision
          Bellman: Q(s,a) = r1 + 0.9 · max Q_target(s', a')
        │
        ▼
[Stage 5] Evaluation
          Pattern Macro F1 · Per-pattern F1 · Mean Halstead delta
          Baselines: Random · Rule-Based · SVM · Supervised-only
          Cross-dataset: MLCQ generalization F1
```

---

## 4. Stage 0 — Data Loading & Preprocessing

**File:** `src/data.py` — `load_smellycode()`

### Fail-Loud Validation (No Fallbacks)

```python
REQUIRED_COLS = {"Code", "GodClass", "FeatureEnvy", "LongMethod", "DataClass",
                 "lloc", "cyclomatic", "V", "D", "E", "B"}
missing = REQUIRED_COLS - set(df.columns)
if missing:
    raise ValueError(f"SmellyCode++ CSV missing columns: {missing}")
```

If any required column is absent, the pipeline crashes immediately with a descriptive error. There is no zero-padding, no default-value fallback, no silent column skip.

### Halstead Normalization Bounds

| Metric Slot | Source Column | Normalization Max | Derivation |
|---|---|---|---|
| idx 3 | `cyclomatic` | 50.0 | 99th percentile of SmellyCode++ train set |
| idx 4 | `lloc` | 5000.0 | 99th percentile of SmellyCode++ train set |
| idx 5 | `V` (Volume) | 5000.0 | 99th percentile |
| idx 6 | `D` (Difficulty) | 100.0 | 99th percentile |
| idx 7 | `E` (Effort) | 1,000,000.0 | 99th percentile |
| idx 8 | `B` (Bugs) | 5.0 | 99th percentile |

### Graph Item Dict (Per Sample)

```python
{
  "x":             Tensor [N, 783],   # N nodes, 783-dim features
  "edge_index":    Tensor [2, E],     # edge connectivity
  "edge_type":     Tensor [E],        # 0=CONTAINS, 1=CALLS, 2=ACCESSES_FIELD
  "y":             Tensor [],         # dominant smell index (0-4)
  "co_smell_mask": Tensor [4],       # binary [GodClass, FeatureEnvy, LongMethod, DataClass]
  "halstead":      Tensor [6],       # raw unnormalized Halstead scalars for reward computation
  "idx":           int,
}
```

---

## 5. Stage 1 — AST Graph Construction

**File:** `src/data.py` — `_build_ast_graph()`

Parses the `Code` column using `javalang`. If parsing fails, raises immediately — no fallback single-node graph.

### Node Types

| Node | One-hot (idx 0-2) | Features |
|---|---|---|
| Class | [1, 0, 0] | Full 6 Halstead + 6 data-usage metrics |
| Method | [0, 1, 0] | Cyclomatic CC, estimated LOC from stmts |
| Field | [0, 0, 1] | Zero CK/Halstead, field declaration embedding |

### Edge Types (Relational)

| Code | Type | Semantics |
|---|---|---|
| 0 | `CONTAINS` | Class ↔ Method, Class ↔ Field (bidirectional) |
| 1 | `CALLS` | Method → Method (intra-class invocation) |
| 2 | `ACCESSES_FIELD` | Method → Field (read/write access) |

R-GAT learns separate attention parameters per edge type — critical for distinguishing FeatureEnvy (many CALLS to external objects) from LongMethod (simply deep CONTAINS nesting).

---

## 6. Stage 2 — NodeFeatureFusion (18.75% Semantic Bottleneck)

**File:** `src/models.py` — `NodeFeatureFusion`

Raw node vectors are 783-dim: 15 structural + 768 semantic. Without fusion, the GCN's first linear layer receives 98.1% semantic signal. NodeFeatureFusion corrects this:

```
Structural branch: 15 → 48 → 104  (2-layer MLP + LayerNorm)
Semantic branch:  768 → 24        (Linear + LayerNorm)
Fused output:     cat([104, 24]) = 128-dim
```

**Semantic weight = 24/128 = 18.75%** — capped at under 20% so structural complexity metrics remain the primary discriminating signal.

### Why 18.75% and Not Higher?

This ratio is validated by Phase 3 of the ablation suite: `SEM_OUT ∈ {8, 16, 24, 32, 48}`. The 18.75% point (SEM_OUT=24) is claimed to maximize minority-class Macro F1. If the sweep shows a different optimum, the paper reports the empirically optimal ratio instead.

---

## 7. Stage 3 — Semantic Embeddings: GraphCodeBERT

**File:** `src/embeddings.py` — `SemanticEmbedder`

Uses `microsoft/graphcodebert-base`. Embeddings are cached to `data/processed/embeddings_cache/<md5>.npy`. If `transformers` is not installed, raises `ImportError` immediately — no synthetic fallback.

**Per-node embedding source:**
- Class node → class identifier name
- Method node → full extracted method body (up to 50 lines)
- Field node → field declaration line

---

## 8. Stage 4 — R-GAT Encoder Architecture

**File:** `src/models.py` — `RGATEncoder`

Converts variable-sized attributed code graphs into a fixed-size **128-dimensional state vector**.

```
Input x  [N, 783]
  │
  └─ NodeFeatureFusion → x_fused [N, 128]
        │
        ├─ RGATLayer 1(128→128): type-specific W_r + attention a_r per edge type
        │   LayerNorm(128)
        │
        ├─ RGATLayer 2(128→128): second relational message passing
        │   skip_proj: Linear(128→128) residual from x_fused
        │   h2 = h2 + skip_proj(x_fused) · LayerNorm(128)
        │
        └─ AttentionPooling: w = sigmoid(Linear(h,1)) / Σw
             z = Σ(w_i · h_i)   → [1, 128]
```

**RGATLayer** learns `num_edge_types=3` separate weight matrices `W_r` and attention vectors `a_r`. A method with many CALLS to external objects (FeatureEnvy signal) gets a different aggregation path from a class with many CONTAINS edges (GodClass/LongMethod signal).

---

## 9. Stage 5 — RL Environment & State Transitions

**File:** `src/environment.py`

### State Transition Function

After the agent selects a refactoring action, the environment simulates the post-refactoring code by perturbing the Halstead/complexity metric dims in the node feature tensor. This produces a genuinely different state `s'` for step 2.

**Perturbation bounds (literature-grounded):**

| Action | Metric Changed | Multiplier Range | Citation |
|---|---|---|---|
| `Facade` | cyclomatic, lloc | ×[0.60, 0.85], ×[0.70, 0.90] | Fowler (1999) §12 |
| `Strategy` | D (Difficulty) | ×[0.70, 0.90] | Fowler (1999) §7 |
| `ExtractClass` | cyclomatic, lloc | ×[0.55, 0.75], ×[0.55, 0.75] | Fowler (1999) §6 |
| `Observer` | D (Difficulty) | ×[0.75, 0.90] | Fowler (1999) §11 |
| `Mediator` | cyclomatic, D | ×[0.70, 0.90], ×[0.80, 0.95] | Brown (1998) §3 |
| `NoRefactor` | — | no change | — |

Unknown action → `ValueError` immediately (no fallback).

These perturbations are **state transition dynamics**, not reported results. The paper reports reward accumulated and Halstead delta — both computed from actual tensor values.

### Soft Reward Matrix (Fowler-Grounded)

| True Smell | Strategy | Observer | Facade | Mediator | ExtractClass | NoRefactor |
|---|---|---|---|---|---|---|
| **GodClass** | -1.0 | -0.5 | **+2.0** | **+1.0** | **+0.5** | -2.0 |
| **FeatureEnvy** | **+2.0** | -1.0 | -1.0 | -0.5 | **+0.5** | -2.0 |
| **LongMethod** | -1.0 | -1.0 | -0.5 | -1.0 | **+2.0** | -2.0 |
| **DataClass** | -1.0 | **+2.0** | -0.5 | -0.5 | -1.0 | -2.0 |
| **NoSmell** | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | **+2.0** |

All base rewards multiplied by `class_weight[smell] = total / (n_classes × count[smell])` — inverse frequency scaling.

### Quality Delta Reward (Step 2 Only)

```
reward_step2 = clip( Σ(halstead_before[3:9] - halstead_after[3:9]) × 2.0, -2.0, +2.0 )
```

Computed from unnormalized `item["halstead"]` tensors. Positive when refactoring reduced complexity/coupling. Negative if it increased them (e.g., unnecessary Facade on clean code).

---

## 10. Stage 6 — DQN Agent Architecture

**File:** `src/models.py` — `SmellDetectionAgent`

```
Input: [1, 128]  (graph embedding from RGATEncoder)
  → Linear(128→256) → ReLU → Dropout(0.2)
  → Linear(256→128) → ReLU → Dropout(0.2)
  → Linear(128→64)  → ReLU → Dropout(0.2)
  → Linear(64→6)              [raw Q-values for 6 refactoring actions]
```

**Target Network:** Periodically synced copy of Q-network with `requires_grad=False`. Synced every `target_update_freq=1000` global steps.

**Hard constraint in `__init__`:**
```python
gamma = cfg["dqn"]["gamma"]
if gamma == 0.0:
    raise ValueError("gamma=0.0 reduces MDP to contextual bandit. Set gamma >= 0.9.")
```

---

## 11. Stage 7 — 2-Step RL Training Loop

**File:** `src/training.py` — `DQNTrainer` (PRIMARY trainer — only trainer)

### Why 2 Steps and Not 1

- With 1 step (γ=0), `Q*(s,a) = r(s,a)` — equivalent to a weighted cross-entropy classifier. No temporal reasoning.
- With 2 steps (γ=0.9), the agent must consider `r1 + 0.9 · max Q_target(s', a')`. The Q-values in step 1 propagate knowledge about what the post-refactoring state looks like — genuine temporal credit assignment.

### Episode Structure

```
Per code sample:
  Step 1: s1 = encode(item)  →  a1 = ε-greedy(Q(s1))  →  r1 = soft_reward + class_weight
          item2 = apply_simulated_refactoring(item, a1)
  Step 2: s2 = encode(item2) →  a2 = ε-greedy(Q(s2), ε/2)  →  r2 = quality_delta_reward
  
  TD target: r1 + γ · max Q_target(s2)
  
  Replay: push (s1,a1,r1,s2,done=False), (s2,a2,r2,s2,done=True)
```

### Replay Buffer & Learning

- FIFO buffer, capacity 50,000 transitions
- Batch size 128, warmup 500 steps before learning begins
- Loss: `SmoothL1Loss(Q(s,a), td_target)` — Huber loss
- Optimizer: Adam, lr=0.0005, weight_decay=1e-5
- LR Schedule: `CosineAnnealingLR(T_max=max_episodes)`
- Gradient clipping: max_norm=1.0

### Epsilon-Greedy Exploration

```
ε(t) = ε_end + (ε_start - ε_end) × max(0, 1 - t / ε_decay_steps)
ε_start=1.0, ε_end=0.05, ε_decay_steps=500,000
```

Decay over 500,000 steps (≈ 150 episodes on SmellyCode++ train set of ~75K samples).

### Logging Per Episode

```
[DQN][Ep N/300] ε=X.XXXX buffer=XXXXXX
[DQN][Ep N][Step M] smell=GodClass
[DQN][Ep N][Step M] a1=Facade r1=2.847
[DQN][Ep N][Step M] a2=NoRefactor r2=0.423
[DQN][Ep N] avg_r1=X.XXX avg_r2=X.XXX loss=X.XXXXXX lr=X.XXXXXX
[DQN][Ep N] val_f1=X.XXXX val_acc=X.XXXX  (every eval_freq=25 episodes)
```

---

## 12. Stage 8 — Evaluation & Baselines

**File:** `src/evaluation.py` — `ExperimentRunner`

### Baselines

All baselines predict **smell class** → mapped to dominant refactoring pattern via `SMELL_TO_PATTERN_DOMINANT = {GodClass:2, FeatureEnvy:0, LongMethod:4, DataClass:1, NoSmell:5}`.

| Baseline | Method |
|---|---|
| Random Agent | Uniformly random pattern (performance floor) |
| Rule-Based | Halstead metric thresholds (cyclomatic>15 → LongMethod, etc.) |
| SVM + Metrics | `SVC(kernel='rbf')` on 6 Halstead features, `StandardScaler` |
| Supervised-only | GCNPretrainer + linear head, no RL fine-tuning |
| **SmellRL (Ours)** | **2-step DQN, γ=0.9, R-GAT + GraphCodeBERT + Halstead** |

### Metrics Reported

| Metric | Description |
|---|---|
| Macro F1 (primary) | Treats all 5 classes equally — penalizes minority class failures |
| Per-pattern F1 | F1 for each of 6 refactoring patterns |
| Accuracy | Overall correct predictions |
| Minority recall | GodClass + FeatureEnvy recall separately |
| Mean cumulative reward | Total reward accumulated on test set (RL-native metric) |
| Mean Halstead delta | Mean improvement in complexity/coupling from correct step-2 transitions |
| Confusion matrix | Saved as CSV in `data/results/` |

### Cross-Dataset Validation

**File:** `scripts/cross_validate_mlcq.py`

After training on SmellyCode++, the trained agent is evaluated on the MLCQ test split (held-out, never used for training). This addresses the "single dataset" threat to validity. Results saved to `data/results/mlcq_generalization.json`.

---

## 13. Ablation Studies (Core Experimental Design)

**File:** `scripts/run_ablations.py`

All 3 phases run through **identical RL training code**. Only the input data or architecture changes — making every comparison a controlled experiment.

### Phase 1 — Semantic Ablation (PRIMARY CLAIM)

| Run | Semantic dims | Graph | Expected Result |
|---|---|---|---|
| `Full` | ✅ GraphCodeBERT (768→24) | ✅ 2-layer R-GAT | Highest Macro F1 |
| `NoSemantic` | ❌ x[:,15:] zeroed | ✅ 2-layer R-GAT | Drops ~5-8% |
| `MetricsOnly` | ❌ x[:,15:] zeroed | ❌ Class node only | Lowest |

**`AblatedDataset`** zeroes dims `[15:783]` — does NOT slice. `sem_proj(zeros) ≈ bias-only ≈ near-zero`. The structural branch is unaffected. Network shape is identical → fair comparison.

### Phase 2 — RL vs Non-RL Comparison

| Run | γ | Training |
|---|---|---|
| `RL_2step` | 0.9 | 2-step MDP DQN (Primary) |
| `RL_bandit` | 0.0 | Contextual bandit DQN |
| `Supervised` | N/A | Linear head on pre-trained encoder only |

Expected: `RL_2step` > `Supervised` > `RL_bandit` on minority-class Macro F1. The RL_bandit underperforms supervised because γ=0 provides no temporal structure and is sample-inefficient vs. dense supervised gradients.

### Phase 3 — Bottleneck Ratio Sweep

`SEM_OUT ∈ {8, 16, 24, 32, 48}` while `STRUCT_OUT = 128 - SEM_OUT`. Plots Macro F1 vs. semantic percentage. Validates or refutes the 18.75% design claim.

### Winner Selection

Phase 1 winner selected on **validation Macro F1** (not test set). Test set is evaluated once, at the end, for final reporting only.

---

## 14. Complete Hyperparameter Table

### Dataset

| Parameter | Value | Justification |
|---|---|---|
| `dataset_type` | `smellycode` | SmellyCode++ primary; MLCQ for cross-validation only |
| `nosmell_ratio` | 3.0 | 3:1 NoSmell:max_smell prevents majority-class bias |
| `multilabel_strategy` | `dominant` | Priority: GodClass > FeatureEnvy > LongMethod > DataClass |
| `train_ratio` | 0.70 | Standard 70/15/15 |
| `random_seed` | 42 | Reproducibility across all splits |

### NodeFeatureFusion

| Parameter | Value | Justification |
|---|---|---|
| `STRUCT_DIM` | 15 | 3 node-type + 6 Halstead + 6 data-usage |
| `SEM_DIM` | 768 | GraphCodeBERT CLS output |
| `STRUCT_OUT` | 104 | 81.25% structural capacity |
| `SEM_OUT` | 24 | 18.75% semantic cap |
| `FUSED_DIM` | 128 | Must equal `gcn.hidden_dim` |

### R-GAT Encoder

| Parameter | Value | Justification |
|---|---|---|
| `hidden_dim` | 128 | Equals FUSED_DIM |
| `output_dim` | 128 | State vector size |
| `num_edge_types` | 3 | CONTAINS, CALLS, ACCESSES_FIELD |
| `num_heads` | 4 | Multi-head attention per edge type |
| `dropout` | 0.1 | Light encoder regularization |
| `use_attention_pool` | true | Learns node importance vs fixed mean |
| `use_residual` | true | Skip connection from x_fused |
| `use_layernorm` | true | Batch-size agnostic normalization |

### GCN Pre-training

| Parameter | Value | Justification |
|---|---|---|
| `epochs` | 100 | Val accuracy plateaus by epoch 80-100 |
| `learning_rate` | 0.001 | 2× RL LR — dense supervised signal |
| `batch_size` | 32 | Per-graph loss accumulation |
| `loss` | `cross_entropy` | Standard supervised pre-training |

### DQN Agent (2-Step RL)

| Parameter | Value | Justification |
|---|---|---|
| `gamma` | **0.9** | Genuine 2-step MDP — **must not be 0** |
| `hidden_layers` | [256, 128, 64] | Compressing bottleneck in Q-network |
| `learning_rate` | 0.0005 | Half of pre-training LR |
| `batch_size` | 128 | Reduces gradient variance |
| `weight_decay` | 1e-5 | L2 regularization |
| `dqn_dropout` | 0.2 | Stronger Q-network regularization |
| `replay_buffer_size` | 50,000 | Breaks temporal correlations |
| `epsilon_start` | 1.0 | Full exploration at start |
| `epsilon_end` | 0.05 | 5% residual exploration |
| `epsilon_decay_steps` | 500,000 | Decays over ~150 episodes on 75K train set |
| `target_update_freq` | 1,000 | Balance stability vs staleness |
| `max_episodes` | 300 | Sufficient for 107K-sample convergence |
| `warmup_steps` | 500 | Minimum buffer fill before learning |
| `eval_freq` | 25 | Val evaluation every 25 episodes |

### Soft Reward Values

| Outcome | Base Value | Justification |
|---|---|---|
| Optimal pattern for smell | +2.0 | Primary objective |
| Acceptable alternative | +0.5 to +1.0 | Partial credit for near-optimal |
| Wrong pattern | -0.5 to -1.0 | Penalizes misaligned choices |
| NoRefactor on smelly code | -2.0 | Highest cost — unaddressed technical debt |

---

## 15. Design Principles

### No Fallbacks — Fail Loudly

Every failure path raises an exception with a precise description of what went wrong and how to fix it. Examples:
- Missing SmellyCode++ column → `ValueError: Missing columns: {X}`
- `gamma=0.0` in config → `ValueError: γ=0.0 reduces to contextual bandit...`
- Pre-train checkpoint missing → `FileNotFoundError: Run --stage pretrain first`
- `transformers` not installed → `ImportError: Install transformers>=4.30.0`

### Log Every Step

Every stage transition, epoch, episode, reward, cache event, and metric is logged at INFO level using `logging.getLogger("SmellRL.<module>")`. Log format:
```
YYYY-MM-DD HH:MM:SS | SmellRL.<module> | LEVEL | [Tag] message
```

### Remove Dead Code

The following were completely removed (not commented out) from v3:
- `SupervisedTrainer` — was primary trainer, replaced by `DQNTrainer`
- `SupervisedSmellDetector` — replaced by `SmellDetectionAgent`
- `SyntheticEmbedder` — no fallback for missing `transformers`
- `GCNEncoder` — superseded by `RGATEncoder`
- `load_mlcq()` — moved to standalone `scripts/cross_validate_mlcq.py`
- `stage_supervised_train()` in main.py
- Phase 2 hyperparameter sweep (focal_gamma/LR) in run_ablations.py

---

## 16. Repository Structure

```
SmellRL/
├── main.py                        # Pipeline entry point (4 stages)
├── config.yaml                    # All hyperparameters
├── requirements.txt
├── Project_info.md                # This document
│
├── src/
│   ├── data.py          # load_smellycode(), AST graph builder, SmellDataset
│   ├── embeddings.py    # SemanticEmbedder (GraphCodeBERT, no fallback)
│   ├── environment.py   # apply_simulated_refactoring(), quality_delta_reward(),
│   │                    # SOFT_REWARD_MATRIX, TRANSITIONS
│   ├── models.py        # NodeFeatureFusion, RGATLayer, RGATEncoder,
│   │                    # DQNClassifier, SmellDetectionAgent
│   ├── training.py      # ReplayBuffer, GCNPretrainer, DQNTrainer (PRIMARY)
│   ├── evaluation.py    # RandomAgent, RuleBasedAgent, SVMAgent,
│   │                    # ExperimentRunner, cross_dataset_eval()
│   └── utils.py         # setup_logger, get_device, CheckpointManager
│
├── scripts/
│   ├── download_smellycode.py   # Figshare download (raises on corrupt file)
│   ├── cross_validate_mlcq.py  # MLCQ generalization evaluation (standalone)
│   └── run_ablations.py        # 3-phase ablation suite
│
├── data/
│   ├── smellycode/              # SmellyCode++.csv
│   ├── mlcq/                    # mlcq.csv (cross-validation only)
│   ├── processed/               # .pt graph cache + embeddings_cache/
│   ├── ablations/               # Per-run ablation outputs
│   └── results/                 # Evaluation CSVs, JSON, confusion matrices
│
└── checkpoints/
    ├── gcn_pretrain_latest.pt   # Warm-started encoder
    └── smellrl_latest.pt        # Full DQN agent (best val F1)
```

---

## 17. Dependencies

| Package | Min Version | Role |
|---|---|---|
| `torch` | ≥2.0.0 | R-GAT, DQN, autograd |
| `transformers` | ≥4.30.0 | GraphCodeBERT tokenizer + model |
| `numpy` | ≥1.24.0 | Array operations |
| `pandas` | ≥1.5.0 | SmellyCode++ CSV loading |
| `scikit-learn` | ≥1.2.0 | Stratified splits, SVM baseline, metrics |
| `pyyaml` | ≥6.0 | Config loading |
| `javalang` | ≥0.13.0 | Java AST parsing |
| `scipy` | ≥1.10.0 | Wilcoxon significance testing |
| `matplotlib` | ≥3.7.0 | Ablation plots |
| `seaborn` | ≥0.12.0 | Confusion matrix heatmaps |
| `tqdm` | ≥4.65.0 | Progress bars |

---

## 18. Reproducibility & Execution Order

### Seeds

- `random_seed=42` for all dataset splits (stratified by `smell_idx`)
- 5-seed significance runs use seeds: 42, 7, 123, 2024, 99

### Statistical Significance

Run all primary comparisons with 5 seeds. Report mean ± std. Apply Wilcoxon signed-rank test (paired, non-parametric) between SmellRL and each baseline. Report p-values in all comparison tables.

### Execution Order

```bash
python scripts/download_smellycode.py           # download SmellyCode++
python main.py --stage preprocess               # build graphs + cache .pt
python main.py --stage pretrain                 # warm-start RGAT encoder
python main.py --stage train                    # 2-step RL (PRIMARY)
python main.py --stage experiment               # baselines + all metrics
python scripts/run_ablations.py --phase 1       # semantic ablation (CORE)
python scripts/run_ablations.py --phase 2       # RL vs supervised ablation
python scripts/run_ablations.py --phase 3       # bottleneck ratio sweep
python scripts/cross_validate_mlcq.py           # MLCQ generalization
```

### Checkpointing

All stages save checkpoints via `CheckpointManager`. Rerunning any command resumes from the latest checkpoint automatically. Stale `.pt` graph caches are detected via `DATA_VERSION` tag and regenerated with a logged warning.

### Compute Budget

- Pre-training: ~45 min, single GPU (4GB VRAM)
- RL training (300 episodes, 75K train samples): ~5-6 hours, single GPU
- Full ablation suite (3 phases × multiple runs): ~24 GPU-hours total
- Minimum hardware: 4GB GPU VRAM (CPU execution ~8× slower)

---

*Document reflects SmellRL v3 as of July 2026.*  
*Sources: src/data.py, src/environment.py, src/models.py, src/training.py, src/evaluation.py, scripts/run_ablations.py, config.yaml*
