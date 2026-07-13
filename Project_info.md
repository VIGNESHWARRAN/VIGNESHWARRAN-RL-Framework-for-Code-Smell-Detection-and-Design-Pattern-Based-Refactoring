# Semantic-Aware SmellRL v3 — Complete Technical Reference
## Resource-Optimized Edition

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

### Resource-Optimized Sub-sampling
To achieve high computational efficiency on commodity hardware (e.g., NVIDIA RTX 3050) without degrading representation robustness:
- **`max_train_samples`**: Cap training set at **5,000 graphs** (stratified).
- **`max_val_samples`**: Cap validation set at **2,000 graphs** (stratified).
- **`max_test_samples`**: Cap test set at **2,000 graphs** (stratified).

Evaluation on 2,000 held-out graphs guarantees a high level of statistical confidence for reported Macro F1 metrics.

### Train / Validation / Test Split

Stratified by `smell_idx` using `sklearn.model_selection.train_test_split`:

| Split | Ratio |
|---|---|
| Train | 70% (capped at 5,000) |
| Val | 15% (capped at 2,000) |
| Test | 15% (capped at 2,000) |

---

## 3. Pipeline Overview

```
SmellyCode++ CSV (107,554 rows)
        │
        ▼
[Stage 0] load_smellycode()
          Multi-label → dominant smell · Halstead metrics parsed
          NoSmell downsampled (3× max smell) · Subsampled to max bounds
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
          RGATEncoder + linear head · Cross-entropy · 40 epochs
          Head discarded · Encoder weights loaded into DQN agent
        │
        ▼
[Stage 4] 2-Step RL Training (PRIMARY — DQN, γ=0.9)
          Step 1: Agent observes code graph → picks refactoring pattern
          Transition: apply_simulated_refactoring() → new metric state
          Step 2: Agent observes new state → refinement decision
          Bellman: Q(s,a) = r1 + 0.9 · max Q_target(s', a') (30 episodes)
        │
        ▼
[Stage 5] Evaluation
          Pattern Macro F1 · Per-pattern F1 · Mean Halstead delta
          Baselines: Random · Rule-Based · SVM
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

#### Concrete Example Trace

```yaml
y: 4                                            # Dominant Smell Index (4 = NoSmell)
co_smell_mask: [0.0, 0.0, 0.0, 0.0]             # Multi-label smells [GodClass, FeatureEnvy, LongMethod, DataClass]
halstead: [1.0, 1.0, 6.34, 0.0, 0.0, 0.002]     # Unnormalized Halstead Metrics: [CC, LLOC, Volume, Difficulty, Effort, Bugs]
x shape: torch.Size([3, 783])                   # 3 AST Nodes, each with 783 dimensions (15 structural + 768 semantic)

x[0, :15] (class node structural features):     # Structural features of class node (features 0 to 14)
  - Node Type (one-hot Class):  [1.0, 0.0, 0.0]
  - Normalized Halstead:        [0.02, 0.0002, 0.0013, 0.0, 0.0, 0.0004] (e.g. 1.0/50.0 = 0.02)
  - AST Data-usage:             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

edge_index shape: torch.Size([2, 4])            # 4 directional connections in the graph
edge_type: [0, 0, 0, 0]                         # Relational edge types (all 4 are type 0 = CONTAINS)
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

---

## 7. Stage 3 — Semantic Embeddings: GraphCodeBERT

**File:** `src/embeddings.py` — `SemanticEmbedder`

Uses `microsoft/graphcodebert-base`. Embeddings are cached to `data/processed/embeddings_cache/<md5>.npy`. If `transformers` is not installed, raises `ImportError` immediately — no synthetic fallback.

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

### Soft Reward Matrix (Fowler-Grounded)

| True Smell | Strategy | Observer | Facade | Mediator | ExtractClass | NoRefactor |
|---|---|---|---|---|---|---|
| **GodClass** | -1.0 | -0.5 | **+2.0** | **+1.0** | **+0.5** | -2.0 |
| **FeatureEnvy** | **+2.0** | -1.0 | -1.0 | -0.5 | **+0.5** | -2.0 |
| **LongMethod** | -1.0 | -1.0 | -0.5 | -1.0 | **+2.0** | -2.0 |
| **DataClass** | -1.0 | **+2.0** | -0.5 | -0.5 | -1.0 | -2.0 |
| **NoSmell** | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | **+2.0** |

All base rewards multiplied by `class_weight[smell]`.

### Quality Delta Reward (Step 2 Only)

```
reward_step2 = clip( Σ(halstead_before[3:9] - halstead_after[3:9]) × 2.0, -2.0, +2.0 )
```

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

---

## 11. Stage 7 — 2-Step RL Training Loop

**File:** `src/training.py` — `DQNTrainer` (PRIMARY trainer)

### Episode Structure

```
Per code sample:
  Step 1: s1 = encode(item)  →  a1 = ε-greedy(Q(s1))  →  r1 = soft_reward + class_weight
          item2 = apply_simulated_refactoring(item, a1)
  Step 2: s2 = encode(item2) →  a2 = ε-greedy(Q(s2), ε/2)  →  r2 = quality_delta_reward
  
  TD target: r1 + γ · max Q_target(s2)
  
  Replay: push (s1,a1,r1,s2,done=False), (s2,a2,r2,s2,done=True)
```

---

## 12. Stage 8 — Evaluation & Baselines

**File:** `src/evaluation.py` — `ExperimentRunner`

### Baselines

All baselines predict **smell class** → mapped to dominant refactoring pattern via `SMELL_TO_PATTERN_DOMINANT`.

| Baseline | Method |
|---|---|
| Random Agent | Uniformly random pattern (performance floor) |
| Rule-Based | Halstead metric thresholds (cyclomatic>15 → LongMethod, etc.) |
| SVM + Metrics | `SVC(kernel='rbf')` on 6 Halstead features, `StandardScaler` |
| **SmellRL (Ours)** | **2-step DQN, γ=0.9, R-GAT + GraphCodeBERT + Halstead** |

### Metrics Reported

- **Macro F1** (primary)
- **Per-pattern F1**
- **Accuracy**
- **Mean cumulative reward** (Total reward accumulated on test set)
- **Mean Halstead delta** (Mean improvement in complexity/coupling)
- **Confusion matrix** (Saved as CSV in `data/results/`)

---

## 13. Ablation Studies (Core Experimental Design)

**File:** `scripts/run_ablations.py`

This suite runs **4 specific core configurations** in sequence to validate the primary scientific claims of the paper (no hyperparameter sweeps).

| Configuration | Description | Key Variable Checked |
|---|---|---|
| **1. `full_rgat`** | **Proposed Method** (DQN 2-Step + RGAT + Halstead + Semantic) | Complete pipeline validation |
| **2. `no_semantic`** | **Semantic Ablation** (DQN 2-Step + RGAT + Halstead) | Verifies the impact of GraphCodeBERT semantics |
| **3. `metrics_only`** | **Graph Ablation** (DQN 2-Step + Flat Halstead metrics only) | Verifies the value of R-GAT graph representations |
| **4. `supervised_only`** | **Paradigm Ablation** (Pre-trained RGAT + Semantic, No RL) | Verifies the value of RL-based reward shaping |

### Configuration Details

1. **`full_rgat` (Proposed Model)**: 
   Fuses the 15 structural metrics (Halstead size/complexity metrics, node-type one-hot, and AST data-usage ratios) with the 768-dimensional GraphCodeBERT CLS embeddings. The agent trains inside the genuine 2-step MDP ($\gamma = 0.9$) utilizing Fowler-grounded reward matrices.

2. **`no_semantic` (Semantic Ablation)**:
   Keeps the R-GAT graph convolution and the 15 structural metrics active, but **zeroes out the 768 semantic dimensions** of the node feature tensor (`x[:, 15:] = 0.0`). The network shapes and parameters are identical to the proposed model. This isolates the exact contribution of Code Language Model semantics.

3. **`metrics_only` (Graph Ablation)**:
   In this configuration, **all graph edges are disconnected** (`edge_index` is empty) and GNN message passing is bypassed. The model only receives the 15-dimensional flat metrics vector for the class node (representing cyclomatic complexity, lines of code, volume, difficulty, effort, and bugs). This isolates the value of modeling the codebase as a relational AST graph.

4. **`supervised_only` (Paradigm Ablation / Normal ML)**:
   Uses the complete RGAT + Semantic input representation, but bypasses the Reinforcement Learning fine-tuning stage. The predictions are generated by attaching a standard linear classifier head directly to the supervised pre-trained encoder weights. This isolates the value of temporal reward-shaping over standard supervised learning.

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
| `data_version` | `"v4"` | Outdates old cached PT files to apply size cap |
| `max_train_samples`| `5000` | Caps training set size for fast run times |
| `max_val_samples`  | `2000` | Caps validation set size for fast evaluation |
| `max_test_samples` | `2000` | Caps test set size for fast evaluation |

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
| `epochs` | **40** | Capped at 40 epochs for fast pre-training |
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
| `epsilon_decay_steps` | **100,000** | Decays over first 20 episodes on capped train split |
| `target_update_freq` | 1,000 | Balance stability vs staleness |
| `max_episodes` | **30** | Capped at 30 episodes for fast convergence |
| `warmup_steps` | 500 | Minimum buffer fill before learning |
| `eval_freq` | **10** | Val evaluation every 10 episodes |

---

## 15. Design Principles

### No Fallbacks — Fail Loudly

Every failure path raises an exception with a precise description of what went wrong.

### Log Every Step

Every stage transition, epoch, episode, reward, cache event, and metric is logged at INFO level.

### Remove Dead Code

Deleted elements: `SupervisedTrainer`, `SupervisedSmellDetector`, `SyntheticEmbedder`, legacy homogeneous `GCNEncoder`, and Phase 3 sweeps.

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
│   ├── download_smellycode.py   # Figshare download
│   ├── cross_validate_mlcq.py  # MLCQ generalization evaluation (standalone)
│   └── run_ablations.py        # Core ablation suite (4 runs)
```

---

## 17. Dependencies

*   `torch >= 2.0.0`
*   `transformers >= 4.30.0`
*   `numpy >= 1.24.0`
*   `pandas >= 1.5.0`
*   `scikit-learn >= 1.2.0`
*   `pyyaml >= 6.0`
*   `javalang >= 0.13.0`
*   `scipy >= 1.10.0`

---

## 18. Reproducibility & Execution Order

### Reproducibility Settings
- `random_seed=42` for all stratified splits.
- Wilcoxon signed-rank test (paired, non-parametric) p-values reported for final comparative results.

### Execution Order

```bash
python scripts/download_smellycode.py           # download SmellyCode++
python main.py --stage preprocess               # build graphs + cache .pt
python scripts/run_ablations.py                 # run 4 core ablation configurations
python scripts/cross_validate_mlcq.py           # MLCQ generalization check
```

### Compute Budget (RTX 3050 GPU)

- **Pre-training (40 epochs)**: **~50 seconds**
- **DQN training (30 episodes)**: **~20 to 25 minutes**
- **Core Ablation Suite (4 runs)**: **~1.5 to 2 hours** (total execution time)

---

*Document reflects SmellRL v3 as of July 2026.*  
*Sources: src/data.py, src/environment.py, src/models.py, src/training.py, src/evaluation.py, scripts/run_ablations.py, config.yaml*
