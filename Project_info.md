# Semantic-Aware SmellRL: Full Project Documentation

**SmellRL** is a hybrid machine-learning pipeline that combines **Graph Convolutional Networks (GCN)**, **Pre-trained Code Language Models (GraphCodeBERT)**, and **Deep Reinforcement Learning (DQN)** to detect structural and semantic code smells in Java source code.

This document is a complete, self-contained technical reference — covering dataset handling, feature engineering, model architecture, training procedures, reward design, evaluation, ablation studies, and all hyperparameters with their justifications.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset: MLCQ](#2-dataset-mlcq)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 0 — Data Loading & Preprocessing](#4-stage-0--data-loading--preprocessing)
5. [Stage 1 — AST Graph Construction & Feature Engineering](#5-stage-1--ast-graph-construction--feature-engineering)
6. [Stage 2 — NodeFeatureFusion (Structural Priority)](#6-stage-2--nodefeaturefusion-structural-priority)
7. [Stage 3 — Semantic Embeddings: GraphCodeBERT](#7-stage-3--semantic-embeddings-graphcodebert)
8. [Stage 4 — GCN Encoder Architecture](#8-stage-4--gcn-encoder-architecture)
9. [Stage 5 — DQN Agent Architecture](#9-stage-5--dqn-agent-architecture)
10. [Stage 6 — RL Training Loop](#10-stage-6--rl-training-loop)
11. [Stage 7 — Evaluation & Baselines](#11-stage-7--evaluation--baselines)
12. [Complete Hyperparameter Table](#12-complete-hyperparameter-table)
13. [Ablation Studies](#13-ablation-studies)
14. [Experimental Results](#14-experimental-results)
15. [Architecture Justification & Design Decisions](#15-architecture-justification--design-decisions)
16. [Repository Structure](#16-repository-structure)
17. [Dependencies](#17-dependencies)
18. [Reproducibility & Resumability](#18-reproducibility--resumability)

---

## 1. Problem Statement

Traditional static code smell detectors (e.g., Designite, PMD) apply hand-coded threshold rules over structural software metrics (CK metrics: WMC, CBO, RFC, LOC, DIT). This approach has two fundamental limitations:

1. **No semantic understanding.** Rules based on `WMC > 47` or `LOC > 1000` cannot distinguish a genuinely complex algorithm from a poorly factored class.
2. **No learning.** Static thresholds cannot adapt to the distribution of smells in a specific codebase or team's style.

SmellRL addresses both limitations by:
- Replacing the metric-only representation with a **heterogeneous AST graph** where every node carries both structural and semantic features from a pre-trained code language model.
- Replacing static threshold rules with a **DQN agent** that learns a classification policy from interaction with labeled examples, guided by an asymmetric reward that encodes domain knowledge about the relative cost of different mistake types.

The target smell taxonomy and their indices are:

| Index | Smell Class   | Description |
|-------|--------------|-------------|
| 0     | `GodClass`   | A class that has taken on too many responsibilities (Blob Anti-Pattern). High WMC, high LOC. |
| 1     | `FeatureEnvy` | A method more interested in data from another class than its own. High external object access. |
| 2     | `LongMethod`  | A method with too many statements, high cyclomatic complexity. |
| 3     | `DataClass`   | A class with many fields but few real methods (anemic domain model). High NOC, low WMC. |
| 4     | `NoSmell`     | Clean, healthy code with no significant structural smells. |

Smell-to-refactoring-pattern mapping:

| Smell       | Valid Refactoring Patterns |
|-------------|--------------------------|
| GodClass    | Facade (2), Mediator (3) |
| FeatureEnvy | Strategy (0), ExtractClass (4) |
| LongMethod  | ExtractClass (4), None (5) |
| DataClass   | Observer (1), None (5) |
| NoSmell     | None (5) |

---

## 2. Dataset: MLCQ

**MLCQ (Machine Learning Code Quality)** is a benchmark dataset of labeled Java classes and methods, curated with human expert review. Each row is a (class/method, smell_type, severity) triple.

- **Source:** CSV file at `data/mlcq/mlcq.csv`
- **Severity labels:** `none`, `minor`, `major`, `critical`. Rows where severity is `"none"` are re-labeled `NoSmell`.
- **Supported smell aliases:** The loader normalises variants: `"godclass"`, `"blob"`, `"god_class"` -> `GodClass`; `"featureenvy"`, `"feature_envy"` -> `FeatureEnvy`; etc.
- **Required columns:** A smell type column (auto-detected from: `smell`, `smell_type`, `kind`, `codesmell`, `type`, `smelltype`) plus a `source_code` column with raw Java source.

### Class Balancing

1. **AST validation:** Using `javalang` via `ProcessPoolExecutor` with `N_CPU - 1` workers, every row's source code is verified to be parseable. Rows that fail are discarded and the validated subset is cached as `<csv>_parseable.csv`.
2. **NoSmell downsampling:** `NoSmell` instances are downsampled to `nosmell_ratio x max_smell_class_count` (default 3.0) to prevent majority-class bias while retaining enough clean-code examples.

### Train / Validation / Test Split

Stratified splits via `sklearn.model_selection.train_test_split`:

| Split | Ratio | Note |
|-------|-------|------|
| Train | 70%   | GCN pre-training and DQN RL training |
| Val   | 15%   | GCN validation and Phase 1 winner selection |
| Test  | 15%   | Held-out for all final evaluations |

`random_seed=42` is used throughout.

---

## 3. Pipeline Overview

```
Raw MLCQ CSV
     |
     v
[Stage 0] Data Loading & Class Balancing
     |  load_mlcq() -> parsed + balanced DataFrame
     |
     v
[Stage 1] AST Graph Construction  (src/data.py)
     |  _build_ast_graph() -> {x: [N, 783], edge_index: [2, E], y: int}
     |  SmellDataset: serialised to data/processed/{train,val,test}.pt
     |
     v
[Stage 2] NodeFeatureFusion  (src/models.py - inside GCNEncoder)
     |  struct[N, 15] -> MLP -> [N, 96]   (75% capacity)
     |  sem[N, 768]   -> Linear -> [N, 32] (25% capacity)
     |  fused: cat([struct_96, sem_32]) = [N, 128]
     |
     v
[Stage 3] GCN Pre-training  (src/training.py -> GCNPretrainer)
     |  GCN + LinearHead trained w/ cross-entropy for 50 epochs
     |  checkpoints/gcn_pretrain_latest.pt
     |
     v
[Stage 4] DQN RL Training  (src/training.py -> DQNTrainer)
     |  Agent observes 128-dim state, takes action in {0..4}
     |  receives asymmetric reward, stores in ReplayBuffer(50k)
     |  checkpoints/smellrl_latest.pt
     |
     v
[Stage 5] Evaluation  (src/evaluation.py -> ExperimentRunner)
         SmellRL vs. Random / Rule-Based / SVM
         data/results/{baseline_comparison,exp5_per_class_f1}.csv
```

---

## 4. Stage 0 — Data Loading & Preprocessing

**File:** `src/data.py` — `load_mlcq()`

### AST Parse Verification

```python
with ProcessPoolExecutor(max_workers=N_CPU-1) as executor:
    results = list(executor.map(check_single_code, codes, chunksize=100))
```

Rows failing `javalang.parse.parse()` are dropped. The validated subset is cached to `<csv>_parseable.csv` — subsequent runs skip the expensive multiprocessing step.

**Why fail-fast, no fallbacks:** A row with unparseable source would silently produce a corrupt graph with zero-valued node features, poisoning the training distribution. Discarding and caching ensures every graph in the dataset corresponds to a genuine, parseable Java class.

---

## 5. Stage 1 — AST Graph Construction & Feature Engineering

**File:** `src/data.py` — `_build_ast_graph()` and `SmellDataset`

Each Java class is parsed into a **heterogeneous directed graph** with three node types:

| Node Type | Source | Index in Graph |
|-----------|--------|---------------|
| Class     | `tree.types[0]`    | Always index 0 |
| Method    | `cls.methods[i]`   | Indices 1 to M |
| Field     | `cls.fields[j]`    | Indices M+1 to M+F |

### Node Feature Vector (783 dims)

Every node carries a **783-dimensional feature vector**:

#### Component 1: Node Type One-Hot (3 dims)

```
[1, 0, 0]  ->  Class node
[0, 1, 0]  ->  Method node
[0, 0, 1]  ->  Field node
```

#### Component 2: Structural / CK Metrics (6 dims, class node)

| Feature | Max Value | Normalization |
|---------|----------|--------------|
| WMC     | 150.0    | clip(val, 0, 150) / 150 |
| DIT     | 8.0      | clip(val, 0, 8) / 8 |
| NOC     | 20.0     | clip(val, 0, 20) / 20 |
| CBO     | 40.0     | clip(val, 0, 40) / 40 |
| RFC     | 200.0    | clip(val, 0, 200) / 200 |
| LOC     | 3000.0   | clip(val, 0, 3000) / 3000 |

Method nodes: `[cc/20, 0, 0, 0, 0, mloc/200]`. Field nodes: all zeros in these slots.

#### Component 3: Data-Usage Features (6 dims, class node)

Computed by traversing method bodies with `javalang.tree.MethodInvocation` and `javalang.tree.MemberReference`:

| Feature | Computation | Purpose |
|---------|-------------|---------|
| `n_field_accesses` | Refs to own fields / 50 | DataClass signal |
| `n_method_calls` | All invocations / 100 | Complexity signal |
| `n_external_objects` | Non-own refs / 50 | FeatureEnvy signal |
| `field_access_ratio` | `n_field / (n_field + n_calls)` | Scale-invariant ratio |
| `method_call_ratio` | `n_calls / (n_field + n_calls)` | Scale-invariant ratio |
| `data_usage_density` | `(n_field + n_calls) / n_methods / 20` | Per-method activity |

#### Component 4: GraphCodeBERT Semantic Embedding (768 dims)

CLS-token embedding from `microsoft/graphcodebert-base`. Cached by MD5 hash of text to `data/processed/embeddings_cache/`.

**Total node feature dimension:** 3 + 6 + 6 + 768 = **783 dims**

### Edge Types

| Edge Type | Definition |
|-----------|-----------|
| `CONTAINS` (structural) | Class <-> each Method; Class <-> each Field (bidirectional) |
| `CALLS` (semantic) | Method A -> Method B if A's body calls B (same class) |
| `ACCESSES_FIELD` (semantic) | Method A -> Field F if A's body references field F |

---

## 6. Stage 2 — NodeFeatureFusion (Structural Priority)

**File:** `src/models.py` — `NodeFeatureFusion`

### The Dimensionality Dominance Problem

In the raw 783-dim node feature vector:
- Semantic (GraphCodeBERT): **768 dims = 98.1%** of the vector
- Structural (CK + data-usage + node-type): **15 dims = 1.9%** of the vector

Without correction, `GCNLayer`'s `nn.Linear(783, 128)` receives 98.1% semantic signal. Gradients for the 15 structural dimensions are proportionally negligible, effectively drowning out the hand-crafted CK and data-usage features.

### Solution: Separate Projection + Late Fusion

`NodeFeatureFusion` projects each group into a balanced space before the GCN ever sees the data:

```
Structural Branch  (15 -> 96):   MLP: Linear(15->48) -> ReLU -> Linear(48->96) -> LayerNorm
Semantic Branch   (768 -> 32):   Linear(768->32) -> LayerNorm
Output:            cat([struct_96, sem_32]) = 128 dims per node
```

| Branch | Input | Output | Capacity |
|--------|-------|--------|----------|
| Structural | 15 dims | 96 dims | **75%** (3x semantic) |
| Semantic   | 768 dims | 32 dims | **25%** |
| Fused      | — | 128 dims | Equal to GCN hidden_dim |

```python
# Class constants
NodeFeatureFusion.STRUCT_DIM = 15   # 3 node-type + 6 CK + 6 data-usage
NodeFeatureFusion.SEM_DIM    = 768  # GraphCodeBERT CLS embedding
NodeFeatureFusion.STRUCT_OUT = 96   # 3x semantic -- structure gets priority
NodeFeatureFusion.SEM_OUT    = 32   # compressed semantic
NodeFeatureFusion.FUSED_DIM  = 128  # STRUCT_OUT + SEM_OUT
```

**Why a 2-layer MLP for structure?** The 15 structural features have non-linear interactions (e.g., high WMC combined with high external object access = FeatureEnvy, not GodClass). A single linear would miss these combinations.

**Why a single linear for semantic?** GraphCodeBERT's 768-dim space is already richly pre-trained — compression just needs to select the most relevant directions. A single linear projection is sufficient.

**Error policy:** If input width < STRUCT_DIM + SEM_DIM, a `ValueError` is raised immediately. The fusion module does NOT silently zero-pad inputs — shape mismatches indicate a pipeline bug and must be caught explicitly.

### Ablation Compatibility

For `ablate_semantic` runs, `AblatedDataset.__getitem__` zeroes dims `[15:]` in-place (via `.clone()` to avoid cache mutation):

```python
item["x"] = item["x"].clone()
item["x"][:, NodeFeatureFusion.STRUCT_DIM:] = 0.0
```

`sem_proj(zeros) ≈ bias-only -> near-zero semantic output`, cleanly disabling the semantic branch without any shape mismatch or silent wrong result.

---

## 7. Stage 3 — Semantic Embeddings: GraphCodeBERT

**File:** `src/embeddings.py` — `SemanticEmbedder`

### Model

`microsoft/graphcodebert-base` — a RoBERTa-based model pre-trained jointly on code and natural language with structural data flow as an auxiliary signal.

### Embedding Strategy

- **Input:** Raw text (class name, method body code, or field declaration)
- **Tokenization:** HuggingFace AutoTokenizer, `max_length=512`, truncated
- **Output:** `last_hidden_state[:, 0, :]` — the `[CLS]` token (shape `[768]`)
- **Caching:** MD5(text) -> `{hash}.npy` in `data/processed/embeddings_cache/`

### Ablation Toggle

```python
USE_EMBEDDINGS_MODEL = True  # src/embeddings.py
```

Setting to `False` routes to `SyntheticEmbedder` — deterministic 768-dim pseudo-embeddings via SHA-256 seeded numpy normals. Used only for explicit structural-only ablation experiments.

**Error policy:** If `USE_EMBEDDINGS_MODEL=True` and `transformers` is not installed, `ImportError` propagates immediately. The system does **not** silently fall back to synthetic embeddings — a missing dependency must be resolved explicitly, not papered over.

---

## 8. Stage 4 — GCN Encoder Architecture

**File:** `src/models.py` — `GCNEncoder`

The GCN encoder converts a variable-size attributed graph into a fixed-size **128-dimensional state vector**:

```
Input x  [N, 783]
  |
  +-- NodeFeatureFusion -> x_fused [N, 128]
  |   (struct 15->96, sem 768->32)
  |
  +-- GCNLayer1(128->128): D^{-1/2} A_hat D^{-1/2} X W + self-loop + ReLU
  |   LayerNorm(128) + Dropout(0.1)
  |
  +-- GCNLayer2(128->128): second round message passing
  |   skip_proj(x_fused): Linear(128->128) residual connection
  |   h2 = h2 + skip_proj(x_fused)
  |   LayerNorm(128)
  |
  +-- AttentionPooling: w = sigmoid(Linear(h,1)) / sum(w)
       z = sum(w_i * h_i)   [1, 128]
```

**GCNLayer** implements: `h = ReLU(W * (D^{-1/2} A_hat D^{-1/2} * X))` where `A_hat = A + I` (self-loops).

**AttentionPooling** learns which nodes (class vs method vs field) contribute most to smell classification via soft attention weights.

**Residual skip connection:** Input `x_fused` (post-fusion, 128 dims) is added to `conv2` output, preventing gradient vanishing and allowing raw structural features to bypass graph convolution if that is more informative.

**GCN Pre-training:** Before RL, the GCN is supervised pre-trained with `CrossEntropyLoss` via a `Linear(128->5)` head for 50 epochs. Only the GCN weights are kept; the head is discarded. This gives the encoder a warm start in a meaningful embedding space before RL begins.

---

## 9. Stage 5 — DQN Agent Architecture

**File:** `src/models.py` — `SmellDetectionAgent`

### DQNClassifier (Q-network)

```
Input: [1, 128]
  -> Linear(128->256) -> ReLU -> Dropout(0.2)
  -> Linear(256->128) -> ReLU -> Dropout(0.2)
  -> Linear(128->64)  -> ReLU -> Dropout(0.2)
  -> Linear(64->5)              [raw Q-values]
```

Hidden layer sizes: `[256, 128, 64]` (compressing bottleneck).

### SmellDetectionAgent

```python
class SmellDetectionAgent(nn.Module):
    gcn:      GCNEncoder        # state encoder (with NodeFeatureFusion)
    q:        DQNClassifier     # online Q-network (gradients flow)
    q_target: DQNClassifier     # target Q-network (frozen, periodically synced)
```

**Target Network:** Initialized as a copy of `q` with `requires_grad=False`. Synced with `q` every `target_update_freq=1000` global steps.

**gamma=0 (Contextual Bandit):** Each code sample is i.i.d. — no temporal dependency. Setting `gamma=0` reduces Bellman to `Q*(s,a) = r(s,a)`, equivalent to a contextual bandit while keeping DQN machinery for future multi-step extensions.

---

## 10. Stage 6 — RL Training Loop

**File:** `src/training.py` — `DQNTrainer`

### Replay Buffer

Circular FIFO, capacity 50,000 transitions: `(state [1,128], action int, reward float)`. Warmup of 100 transitions before training begins. Mini-batch size: 128.

### Epsilon-Greedy Exploration

```
epsilon(t) = epsilon_end + (epsilon_start - epsilon_end) * max(0, 1 - t / epsilon_decay_steps)
```

Start=1.0, End=0.05, Decay=250,000 steps.

### Reward Function (Asymmetric, Class-Weighted)

| Outcome | Base Reward |
|---------|------------|
| Correct smell detected | +2.0 |
| Correct NoSmell | +0.5 |
| Wrong smell type | -1.0 |
| False alarm (smell on clean code) | -0.5 |
| **Missed smell (NoSmell on smelly)** | **-2.0** |

Base reward multiplied by `weight[cls] = total_samples / (n_classes * count[cls])` — inverse class frequency. Rare smell classes yield proportionally larger rewards when correctly identified.

**Asymmetry rationale:** In software quality contexts, missing a real smell (allowing technical debt to accumulate) is costlier than raising a false alarm. The `-2.0` penalty for missed smells reflects this domain knowledge.

### Loss Function & Optimization

```
target = reward   (gamma=0, no bootstrap)
loss   = SmoothL1Loss(Q(s, a), target)   # Huber loss
```

- **Optimizer:** Adam, `lr=0.0005`, `weight_decay=1e-5`
- **LR Schedule:** `CosineAnnealingLR(T_max=max_episodes)` — smooth 0.0005->0 decay
- **Gradient clipping:** `max_norm=1.0`
- **Joint optimization:** GCN encoder and Q-network are optimized together — GCN is not frozen during RL

---

## 11. Stage 7 — Evaluation & Baselines

**File:** `src/evaluation.py` — `ExperimentRunner`

### Random Agent

Uniformly random smell class prediction. Performance floor.

### Rule-Based Agent (Designite-like)

Hand-coded CK metric thresholds on raw node features:

```python
# reads from item["x"] - raw [N, 783] node feature tensor
wmc = cf[3] * 150.0    # index 3 = WMC normalized
cbo = cf[6] * 40.0     # index 6 = CBO normalized
rfc = cf[7] * 200.0    # index 7 = RFC normalized
loc = cf[8] * 3000.0   # index 8 = LOC normalized
# Method/field counts from node-type one-hot flags (correct implementation):
n_m = int(x[:, 1].sum().item())   # x[:, 1] = is_method flag
n_f = int(x[:, 2].sum().item())   # x[:, 2] = is_field flag

if wmc > 47 or loc > 1000:             -> GodClass
if cbo > 10 and rfc > 50:              -> FeatureEnvy
if (loc / n_m) > 100 or avg_cc > 8:   -> LongMethod
if n_f > 8 and wmc < 10:              -> DataClass
else:                                  -> NoSmell
```

Note: `n_m` and `n_f` are computed from the one-hot feature flags (`x[:, 1]` and `x[:, 2]`), not from `(N-1)//2` estimation.

### SVM + CK Metrics

`sklearn.svm.SVC(kernel='rbf', C=1.0)` on 6 CK metrics (indices 3-8 of class node features), standardized with `StandardScaler`. Learned metric-only baseline.

### Metrics

Macro Precision, Macro Recall, Macro F1 (primary), Accuracy, per-class F1, confusion matrix.

---

## 12. Complete Hyperparameter Table

### Dataset

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `train_ratio` | 0.70 | Standard 70/15/15 |
| `val_ratio` | 0.15 | Model selection |
| `test_ratio` | 0.15 | Final evaluation |
| `random_seed` | 42 | Reproducibility |
| `nosmell_ratio` | 3.0 | 3:1 NoSmell:max_smell — prevents majority-class bias |

### NodeFeatureFusion

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `STRUCT_DIM` | 15 | 3 node-type + 6 CK + 6 data-usage |
| `SEM_DIM` | 768 | GraphCodeBERT output |
| `STRUCT_OUT` | 96 | 3x semantic capacity (75% of fused dim) |
| `SEM_OUT` | 32 | Compressed semantic (25% of fused dim) |
| `FUSED_DIM` | 128 | Must equal GCN hidden_dim |
| MLP layers (struct) | 15->48->96 | Non-linear feature interaction learning |
| Linear (sem) | 768->32 | Single projection — semantics already pre-trained |

### Semantic Embeddings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `model_name` | `microsoft/graphcodebert-base` | Pre-trained on code + graph structure |
| `embedding_dim` | 768 | Fixed by GraphCodeBERT |
| `max_length` | 512 | ~95% of method bodies fit within 512 BPE tokens |
| `use_cache` | `true` | Embedding is expensive; reuse across runs |

### GCN Encoder

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `hidden_dim` | 128 | Equals FUSED_DIM — no dimension change after fusion |
| `output_dim` | 128 | State vector size |
| `num_layers` | 2 | 2-hop neighborhood; more layers risk over-smoothing |
| `dropout` | 0.1 | Light regularization |
| `use_attention_pool` | true | Learns node importance vs fixed mean/max |
| `use_residual` | true | Skip connection from x_fused to conv2 output |
| `use_layernorm` | true | Batch-size agnostic |

### GCN Pre-training

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `epochs` | 50 | Val accuracy plateaus by epoch 40-50 |
| `learning_rate` | 0.001 | 2x RL LR — dense supervised signal |
| `batch_size` | 32 | Accumulated per-graph losses |
| `checkpoint_freq` | 10 | Recovery within 10 epochs |

### DQN Agent

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `hidden_layers` | `[256, 128, 64]` | Compressing bottleneck |
| `learning_rate` | 0.0005 | Half of pre-training LR |
| `batch_size` | 128 | Large batch reduces gradient variance |
| `weight_decay` | 1e-5 | L2 regularization |
| `dqn_dropout` | 0.2 | Stronger regularization on Q-network |
| `replay_buffer_size` | 50,000 | ~50 episodes; breaks temporal correlations |
| `epsilon_start` | 1.0 | Full exploration at start |
| `epsilon_end` | 0.05 | 5% residual exploration |
| `epsilon_decay_steps` | 250,000 | Decays over ~125 episodes |
| `target_update_freq` | 1,000 | Balance stability vs staleness |
| `max_episodes` | 200 | Sufficient for convergence on MLCQ |
| `warmup_steps` | 100 | Minimum buffer fill before learning |
| `gamma` | 0.0 | Single-step contextual bandit |

### Reward Shaping

| Outcome | Value | Justification |
|---------|-------|---------------|
| `correct_smell` | +2.0 | Primary objective |
| `correct_nosmell` | +0.5 | Beneficial but less critical |
| `incorrect_smell` | -1.0 | Wrong type; less bad than complete miss |
| `false_alarm` | -0.5 | Low stakes compared to missed smells |
| `missed_smell` | -2.0 | Highest cost — technical debt miss |

---

## 13. Ablation Studies

**File:** `scripts/run_ablations.py`

### Phase 1: Hyperparameter Sweep

Three full-architecture runs with different RL hyperparameters:

| Run Name | LR | epsilon_decay | correct_smell | missed_smell |
|----------|----|--------------|--------------|--------------|
| `phase1_slow_explorer` | 0.0005 | 250,000 | +2.0 | -1.0 |
| `phase1_cautious_learner` | 0.0001 | 150,000 | default | default |
| `phase1_strict_evaluator` | 0.0005 | 150,000 | +2.0 | -3.0 |

Winner = highest validation Macro F1 -> hyperparameters inherited by Phase 2.

### Phase 2: Component Ablations

| Ablation | GCN Type | Semantic | Description |
|----------|----------|----------|-------------|
| `ablate_semantic` | Full 2-layer | **Disabled** | x[:, 15:] zeroed -> struct-only fusion |
| `ablate_structure` | **1-layer GCN** | Enabled | Shallow graph, 1-hop only |
| `ablate_traditional` | **1-layer GCN** | **Disabled** | Struct-only + shallow graph |

### Ablation Modules

**`AblatedDataset`** — zeroes semantic dims (does NOT slice):

```python
item["x"] = item["x"].clone()
item["x"][:, NodeFeatureFusion.STRUCT_DIM:] = 0.0
# sem_proj(zeros) ~ bias-only => near-zero semantic output
```

**`OneLayerGCN`** — 1-layer message passing + mean pooling. `conv1` set to `GCNLayer(FUSED_DIM, out_ch)` since fusion is inherited from `GCNEncoder`.

**`IdentityGCN`** — Applies fusion then operates on class node only (no message passing):

```python
def forward(self, x, edge_index):
    x = self.fusion(x)        # fusion still applied
    h = x[0:1]                # class node only
    h = F.relu(self.conv1.linear(h))
    h = self.conv2.linear(h)
    return h
```

---

## 14. Experimental Results

### Baseline Comparison (Phase 1 Slow Explorer, with Embeddings)

| Method | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|--------|----------|----------|----------------|--------------|
| Random Agent | 0.1998 | 0.1791 | 0.1976 | 0.2013 |
| Rule-Based (Designite-like) | 0.3687 | 0.1555 | 0.1398 | 0.1912 |
| SVM + CK Metrics | 0.5023 | 0.1337 | 0.1005 | 0.2000 |
| **SmellRL (Ours)** | **0.5034** | **0.4585** | **0.4493** | **0.5079** |

### Per-Class F1 Breakdown

| Smell Type | SmellRL F1 | Rule-Based F1 | Delta |
|------------|-----------|--------------|-------|
| GodClass    | 0.4717    | 0.0000       | +0.4717 |
| FeatureEnvy | 0.2013    | 0.0000       | +0.2013 |
| LongMethod  | 0.5236    | 0.0000       | +0.5236 |
| DataClass   | 0.5374    | 0.1788       | +0.3586 |
| NoSmell     | 0.5586    | 0.5989       | -0.0403 |

### SmellRL Confusion Matrix (Test Set)

|              | GodClass | FeatureEnvy | LongMethod | DataClass | NoSmell |
|--------------|----------|------------|-----------|----------|--------|
| **GodClass** | 75 | 1  | 2  | 36 | 17  |
| **FeatureEnvy** | 0 | 15 | 29 | 1  | 13  |
| **LongMethod**  | 2 | 32 | 61 | 0  | 5   |
| **DataClass**   | 27 | 0 | 0  | 97 | 23  |
| **NoSmell**     | 83 | 43 | 41 | 80 | 193 |

---

## 15. Architecture Justification & Design Decisions

### Why NodeFeatureFusion with 3:1 structural priority?

The raw 783-dim vector has 98.1% semantic content. Without fusion, `GCNLayer`'s `nn.Linear(783, 128)` effectively discards the 15 structural dimensions since their gradient signal is proportionally negligible. NodeFeatureFusion solves this by:
1. Giving structural features a 2-layer MLP to learn non-linear interactions (WMC x data-usage = FeatureEnvy distinction)
2. Compressing GraphCodeBERT's 768-dim to a compact 32-dim slot (rich pre-training; compression suffices)
3. Achieving a 3:1 capacity ratio (96:32) reflecting that hand-crafted structural metrics are more directly diagnostic of smell types than general semantic code embeddings

### Why graph-based representation instead of sequence models?

Java classes have a fundamentally graph-structured form: methods call other methods, methods access fields. A GCN can propagate information along `CALLS` edges — a short method that delegates to many long ones can still be detected as contributing to LongMethod complexity. Flat-sequence BERT models lose these structural relationships entirely.

### Why pre-train the GCN before RL?

RL is sample-inefficient — learning both a graph representation AND an optimal policy simultaneously from sparse reward with ~1,000-3,000 examples is extremely difficult. GCN pre-training with dense cross-entropy supervision gives the encoder a warm start in a semantically meaningful embedding space. The RL phase then only fine-tunes the decision boundary.

### Why DQN instead of supervised learning throughout?

1. **Asymmetric cost modeling:** The `missed_smell=-2.0` >> `false_alarm=-0.5` reward encodes domain knowledge that cross-entropy loss cannot express without manual class weights.
2. **Extensibility:** DQN naturally extends to multi-step sequential refactoring decisions (observe smell -> select pattern -> observe quality metric improvement).
3. **Class imbalance handling:** Inverse-frequency reward scaling naturally addresses class imbalance without manual loss weight tuning.

### Why gamma=0 (Contextual Bandit)?

Each code sample is i.i.d. — there is no temporal dependency between classifying one code unit and the next. `gamma=0` makes `Q*(s,a) = r(s,a)`, equivalent to a contextual bandit. The full DQN machinery (target network, replay buffer) is maintained for future multi-step extensions.

### Why fail-loud instead of silent fallbacks?

Three specific fallbacks were removed from the codebase:
1. **`transformers` ImportError** (`src/embeddings.py`): Previously silently fell back to hash-based synthetic embeddings. Now raises `ImportError`. A missing dependency must be resolved explicitly.
2. **`NodeFeatureFusion` size mismatch** (`src/models.py`): Previously zero-padded silently. Now raises `ValueError`. A shape mismatch indicates a pipeline bug.
3. **`is_graph_virtual()`** (`scripts/run_ablations.py`): Removed entirely. Virtual graphs were a relic of an older pipeline design where AST parsing failures generated fallback graphs. Since all rows now fail loudly at parse validation, virtual graphs cannot exist.

---

## 16. Repository Structure

```
SmellRL/
├── main.py                        # Pipeline entry point (4 stages)
├── config.yaml                    # All hyperparameters and paths
├── requirements.txt               # Python dependencies
├── README.md                      # Quick-start guide
├── Project_info.md                # This document
│
├── src/
│   ├── data.py        # MLCQ loading, AST parsing, graph builder, SmellDataset
│   ├── embeddings.py  # GraphCodeBERT wrapper + SyntheticEmbedder (ablation only)
│   ├── models.py      # NodeFeatureFusion, GCNLayer, GCNEncoder, DQNClassifier, SmellDetectionAgent
│   ├── training.py    # ReplayBuffer, GCNPretrainer, DQNTrainer
│   ├── evaluation.py  # RandomAgent, RuleBasedAgent (fixed n_m/n_f), SVMAgent, ExperimentRunner
│   └── utils.py       # setup_logger, get_device, CheckpointManager
│
├── scripts/
│   ├── code_download.py     # Dataset download utility
│   └── run_ablations.py     # 2-phase ablation runner
│                            # AblatedDataset (zero-out, not slice)
│                            # OneLayerGCN, IdentityGCN
│
├── diagrams/
│   ├── data_pipeline&feature_extract.html   # Stage 0-2 pipeline (no fallback arrows)
│   ├── graph_encoder.html                   # GCN encoder with NodeFeatureFusion
│   ├── rl_decision_agent.html               # DQN agent with reward table
│   └── overall_architecture.html            # 5-phase end-to-end overview
│
├── data/
│   ├── mlcq/                # Raw MLCQ CSV
│   ├── processed/           # Serialized .pt graph files (783-dim node features)
│   │   └── embeddings_cache/    # GraphCodeBERT .npy MD5-keyed cache
│   └── results/             # Evaluation CSVs and JSON outputs
│
└── checkpoints/             # Model checkpoints
    ├── gcn_pretrain_latest.pt   # GCN encoder after supervised pre-training
    └── smellrl_latest.pt        # Full agent (GCN+NodeFeatureFusion+DQN)
```

---

## 17. Dependencies

| Package | Min Version | Role |
|---------|------------|------|
| `torch` | >=2.0.0 | Tensor ops, GCN/DQN implementation, autograd |
| `transformers` | >=4.30.0 | GraphCodeBERT tokenizer + model |
| `numpy` | >=1.24.0 | Array ops, SyntheticEmbedder |
| `pandas` | >=1.5.0 | MLCQ CSV loading |
| `scikit-learn` | >=1.2.0 | Stratified splits, SVM baseline, metrics |
| `pyyaml` | >=6.0 | Config loading |
| `javalang` | >=0.13.0 | Java AST parsing |
| `matplotlib` | >=3.7.0 | Plotting |
| `seaborn` | >=0.12.0 | Statistical plotting |
| `tqdm` | >=4.65.0 | Progress bars |
| `scipy` | >=1.10.0 | Statistical testing |

---

## 18. Reproducibility & Resumability

### Seeds

- `random_seed=42` for all dataset splits (stratified by `smell_idx`)
- `SyntheticEmbedder` uses SHA-256 -> seed -> `np.random.RandomState` for deterministic ablation embeddings

### Checkpoint System

Every stage saves checkpoints via `CheckpointManager`:
- Full `state_dict()` of all models (GCN + NodeFeatureFusion + DQN)
- Optimizer state, epsilon value, training history

Re-running any command resumes from the latest checkpoint:
```bash
python main.py                      # resumes all stages
python main.py --stage pretrain     # GCN supervised pre-training only
python main.py --stage train        # DQN RL training only
python main.py --stage experiment   # evaluation only
python scripts/run_ablations.py     # 2-phase ablation (isolated checkpoints per run)
python scripts/run_ablations.py --test  # dry-run with tiny slice
```

### 3-Layer Data Caching

1. `load_mlcq()` caches parseable rows to `<csv>_parseable.csv`
2. `run_preprocessing()` checks for existing `train.pt / val.pt / test.pt`
3. `SemanticEmbedder.embed_identifier()` checks `embeddings_cache/<md5>.npy`

After the first full run, all subsequent runs (including all ablation sweeps) reuse all cached data.

---

*Document reflects SmellRL implementation as of July 2026.*
*Sources: src/data.py, src/models.py, src/training.py, src/evaluation.py, src/embeddings.py, scripts/run_ablations.py, config.yaml*
