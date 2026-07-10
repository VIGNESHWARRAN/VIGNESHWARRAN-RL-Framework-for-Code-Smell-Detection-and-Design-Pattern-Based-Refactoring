# Semantic-Aware SmellRL: Full Project Documentation

**SmellRL** (v2: Refactoring Edition) is a hybrid reinforcement learning pipeline that combines **Relational Graph Attention Networks (R-GAT)**, **Pre-trained Code Language Models (GraphCodeBERT)**, and **Deep Reinforcement Learning (DQN)** to directly recommend software refactoring design patterns (`Strategy`, `Observer`, `Facade`, `Mediator`, `ExtractClass`, `None`) on Java source code.

This document is a complete, self-contained technical reference — covering dataset handling, feature engineering, model architecture, training procedures, reward design, evaluation, ablation studies, and all hyperparameters with their justifications.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset: MLCQ](#2-dataset-mlcq)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 0 — Data Loading & Preprocessing](#4-stage-0--data-loading--preprocessing)
5. [Stage 1 — AST Graph Construction & Feature Engineering](#5-stage-1--ast-graph-construction--feature-engineering)
6. [Stage 2 — NodeFeatureFusion (18.75% Semantic Bottleneck)](#6-stage-2--nodefeaturefusion-1875-semantic-bottleneck)
7. [Stage 3 — Semantic Embeddings: GraphCodeBERT](#7-stage-3--semantic-embeddings-graphcodebert)
8. [Stage 4 — GCN/R-GAT Encoder Architecture](#8-stage-4--gcnr-gat-encoder-architecture)
9. [Stage 5 — DQN Agent Architecture](#9-stage-5--dqn-agent-architecture)
10. [Stage 6 — RL Training Loop & Soft Rewards](#10-stage-6--rl-training-loop--soft-rewards)
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

Traditional code smell detectors (e.g., Designite, PMD) apply hand-coded threshold rules over structural metrics, which lacks semantic understanding. Furthermore, treating refactoring as a simple classification task over smells fails because software design is not binary—multiple refactoring choices exist with varying degrees of correctness.

SmellRL addresses these limitations by:
- Bypassing the explicit bottleneck of smell detection to directly recommend **Refactoring Design Patterns** (`Strategy`, `Observer`, `Facade`, `Mediator`, `ExtractClass`, `None`).
- Formulating refactoring recommendation as a **Contextual Bandit with Soft Rewards**, allowing the model to learn from degrees of correctness (e.g., recommending `Facade` is highly optimal for a `GodClass`, but `Mediator` is also an acceptable alternative).
- Fusing hand-crafted AST structural metrics with high-dimensional pre-trained **GraphCodeBERT embeddings** using a restricted bottleneck to guarantee a balance of structural and semantic signals.

The target Refactoring action space and their indices are:

| Index | Refactoring Pattern | Rationale / Target Smell |
|---|---|---|
| 0 | `Strategy` | Isolates feature-envious methods |
| 1 | `Observer` | Decouples data-centric classes |
| 2 | `Facade` | Unifies complex interface entrypoints for Blobs |
| 3 | `Mediator` | Reduces structural class coupling for Blobs |
| 4 | `ExtractClass` | Extracts long methods or splits bloated classes |
| 5 | `None` | Kept for clean code |

---

## 2. Dataset: MLCQ

**MLCQ (Machine Learning Code Quality)** is a benchmark dataset of labeled Java classes and methods, curated with human expert review. Each row is a (class/method, smell_type, severity) triple.

- **Source:** CSV file at `data/mlcq/mlcq.csv`
- **Severity labels:** `none`, `minor`, `major`, `critical`. 
- **Severity Filter (v2):** Rows labeled with `minor` severity are discarded to remove label noise and sharpen the decision boundaries of the models.
- **Supported smell aliases:** The loader normalises variants: `"godclass"`, `"blob"`, `"god_class"` -> `GodClass`; `"featureenvy"`, `"feature_envy"` -> `FeatureEnvy`; etc.

### Class Balancing

1. **AST validation:** Using `javalang` via `ProcessPoolExecutor` with `N_CPU - 1` workers, every row's source code is verified to be parseable. Rows that fail are discarded and the validated subset is cached as `<csv>_parseable.csv`.
2. **NoSmell downsampling:** `NoSmell` instances are downsampled to `nosmell_ratio x max_smell_class_count` (default 3.0) to prevent majority-class bias while retaining enough clean-code examples.

### Train / Validation / Test Split

Stratified splits via `sklearn.model_selection.train_test_split`:

| Split | Ratio | Note |
|-------|-------|------|
| Train | 70%   | Supervised encoder pre-training and DQN RL training |
| Val   | 15%   | Hyperparameter validation and Phase 1 winner selection |
| Test  | 15%   | Held-out for final evaluations and baselines comparisons |

---

## 3. Pipeline Overview

```
Raw MLCQ CSV
     |
     v
[Stage 0] Data Loading & Severity Filtering (drops minor rows)
     |
     v
[Stage 1] AST Graph Construction & Embeddings (783-dim node features)
     |
     v
[Stage 2] Task Transfer Pre-training (trains GCN/R-GAT encoder on smell classification)
     |
     v
[Stage 3] Direct-RL Refactoring Training (DQN with Soft Rewards mapping to refactoring patterns)
     |
     v
[Stage 4] Evaluation & Simulated Refactoring Quality Impact Analysis
```

---

## 4. Stage 0 — Data Loading & Preprocessing

**File:** `src/data.py` — `load_mlcq()`

Loads the CSV, cleans column headers, normalises smell labels, filters out `minor` severity rows, and performs stratified downsampling of the majority class (`NoSmell`). DISCARDS code that does not successfully parse into a `javalang` AST tree.

---

## 5. Stage 1 — AST Graph Construction & Feature Engineering

**File:** `src/data.py` — `_build_ast_graph()`

Constructs a heterogeneous graph from Java class declarations. 
Nodes are typed as:
- Class Node (index 0)
- Method Nodes
- Field Nodes

### Structural Features (15 dimensions)
Every node starts with a 15-dimensional structural vector:
1. **Node Type Indicators (3 dims):** One-hot representation of `[is_class, is_method, is_field]`.
2. **CK Metrics (6 dims):** Normalized WMC, DIT, NOC, CBO, RFC, LOC. Normalized via `_norm` using empirical maximum bounds.
3. **Data Usage Metrics (6 dims):** Computes `n_field_accesses`, `n_method_calls`, `n_external_objects`, `field_access_ratio`, `method_call_ratio`, and `data_usage_density`.

### Relational Edges
Unlike legay homogenous GCNs, SmellRL v2 models relational structures via three edge types:
- `CONTAINS` (0): Bidirectional connection between class node and method/field nodes.
- `CALLS` (1): Directed connection mapping method-to-method calls (intra-class).
- `ACCESSES_FIELD` (2): Directed connection mapping method-to-field read/writes.

---

## 6. Stage 2 — NodeFeatureFusion (18.75% Semantic Bottleneck)

**File:** `src/models.py` — `NodeFeatureFusion`

Concatenating structural features (15 dims) and GraphCodeBERT embeddings (768 dims) creates a 783-dim vector where 98.1% of the signal is semantic. To prevent the semantic embeddings from drowning out structural metrics, we apply a fused bottleneck:
1. **Structural Branch:** Passes the 15 metrics through a 2-layer MLP (`15 -> 48 -> 104`) to learn complex, non-linear interactions.
2. **Semantic Branch:** Passes the 768-dim embeddings through a linear layer and LayerNorm (`768 -> 24`) to compress the redundant high-dimensional spaces.
3. **Concat:** Concatenates both outputs `[104 | 24]` to construct a balanced **128-dimensional** node state.
4. **Semantics Weightage:** The semantic representation is strictly capped at **18.75%** (`24 / 128 = 18.75%`).

---

## 7. Stage 3 — Semantic Embeddings: GraphCodeBERT

**File:** `src/embeddings.py` — `SemanticEmbedder`

Extracts semantic embeddings for code fragments (full method bodies and field declarations) using the `microsoft/graphcodebert-base` Transformer model.
- Uses `embeddings_cache/` directory to store MD5-keyed `.npy` files of representations, accelerating consecutive training iterations.
- If semantic branch is ablated (`scripts/run_ablations.py`), the semantic indices `[15:783]` are zeroed out via `AblatedDataset`.

---

## 8. Stage 4 — GCN/R-GAT Encoder Architecture

**File:** `src/models.py` — `RGATEncoder` / `GCNEncoder`

Converts variable-sized attributed code graphs into a fixed-size **128-dimensional state vector**.

SmellRL v2 defaults to a **Relational Graph Attention Network (R-GAT)**:
- Standard GCNs aggregate all neighbors equally. In software, this is counterproductive (a method calling 10 external classes signals FeatureEnvy, but a class containing 10 methods is normal).
- R-GAT Layer learns distinct weight matrices $W_r$ and attention parameters $a_r$ for each edge type (`CONTAINS`, `CALLS`, `ACCESSES_FIELD`).
- Memory-efficient projection is performed globally before scattering to edges to avoid CUDA Out-Of-Memory errors.
- Attention pooling is used at the graph readout layer to learn node-level importances.

---

## 9. Stage 5 — DQN Agent Architecture

**File:** `src/models.py` — `SmellDetectionAgent` (repurposed for refactoring)

Wraps the R-GAT/GCN encoder and maps the 128-dim graph embedding to Q-values for the 6 Refactoring actions:
```
Input: [1, 128]
  -> Linear(128->256) -> ReLU -> Dropout(0.2)
  -> Linear(256->128) -> ReLU -> Dropout(0.2)
  -> Linear(128->64)  -> ReLU -> Dropout(0.2)
  -> Linear(64->6)              [raw Q-values for 6 actions]
```

---

## 10. Stage 6 — RL Training Loop & Soft Rewards

**File:** `src/training.py` — `DQNTrainer`

Triggers a Contextual Bandit episodic loop ($\gamma = 0$):
- **Replay Buffer:** FIFO buffer (capacity 50,000 transitions), sampled in batches of 128.
- **Epsilon Greedy:** Decay exploration rate from 1.0 to 0.05 over 250,000 steps.
- **Task Transfer Pre-training:** To solve sample-inefficiency, the agent's encoder is pre-trained to classify the 5 raw code smells (supervised cross-entropy/focal loss). Once encoder pre-training completes, the classification head is dropped, the weights are frozen/loaded into the DQN agent, and the agent trains on the refactoring bandit task.

### Soft Reward Matrix
Instead of evaluating refactoring as a strict 1-to-1 binary label, the trainer uses a Soft Reward Matrix based on the true underlying smell of the code:

| True Smell | Strategy (0) | Observer (1) | Facade (2) | Mediator (3) | ExtractClass (4) | None (5) |
|---|---|---|---|---|---|---|
| **GodClass** (0) | -1.0 | -1.0 | **+2.0** | **+1.0** | **+0.5** | -2.0 |
| **FeatureEnvy** (1) | **+2.0** | -1.0 | -1.0 | -1.0 | **+1.0** | -2.0 |
| **LongMethod** (2) | -1.0 | -1.0 | -1.0 | -1.0 | **+2.0** | -2.0 |
| **DataClass** (3) | -1.0 | **+2.0** | -1.0 | -1.0 | -1.0 | -2.0 |
| **NoSmell** (4) | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | **+2.0** |

All base rewards are scaled by the dataset inverse-frequency weights to ensure minority smell configurations aren't neglected.

---

## 11. Stage 7 — Evaluation & Baselines

**File:** `src/evaluation.py` — `ExperimentRunner`

Evaluates the agent against SVM, Rule-Based, and Random baselines in the Refactoring Pattern space.
- Baselines predict the smell class, which is mapped to the dominant refactoring pattern index (e.g. GodClass -> Facade) via `SMELL_TO_PATTERN_DOMINANT`.
- Performs macro precision/recall/F1 evaluations, per-pattern breakdowns, and confusion matrix generations.

### Simulated Code Quality Impact metrics
To show that the refactorings actually improve code quality, we simulate the changes on the code metrics:
1. **WMC Complexity Reduction (%)**: Valid refactorings on `GodClass` or `LongMethod` reduce complexity (WMC) by **15% to 40%**.
2. **CBO Coupling Reduction (%)**: Valid refactorings on `FeatureEnvy` or `DataClass` reduce coupling (CBO) by **10% to 30%**.
3. **LOC Size Reduction (%)**: Valid refactorings on `LongMethod` reduce class size parameters by **30%**.
4. **False Alarm Penalty**: If the agent unnecessarily refactors `NoSmell` code, CBO is increased by **15%** to simulate delegation/wrapper structural overhead.

Metrics are saved to `baseline_comparison.csv` under:
- `WMC_Complexity_Reduction(%)`
- `CBO_Coupling_Reduction(%)`
- `LOC_Size_Reduction(%)`

---

## 12. Complete Hyperparameter Table

### Representation & Fusion
- `STRUCT_DIM`: 15
- `SEM_DIM`: 768
- `STRUCT_OUT`: 104 (81.25% weightage)
- `SEM_OUT`: 24 (18.75% weightage)
- `FUSED_DIM`: 128
- MLP Layers (Structural): 15 -> 48 -> 104

### R-GAT Encoder
- `hidden_dim`: 128
- `output_dim`: 128
- `num_edge_types`: 3
- `num_heads`: 4
- `dropout`: 0.1
- `use_attention_pool`: True
- `use_residual`: True
- `use_layernorm`: True

### DQN Optimizer & Agent
- `hidden_layers`: `[256, 128, 64]`
- `learning_rate`: 0.0005
- `batch_size`: 128
- `weight_decay`: 1e-5
- `dqn_dropout`: 0.2
- `replay_buffer_size`: 50,000
- `epsilon_decay_steps`: 250,000
- `target_update_freq`: 1,000

---

## 13. Ablation Studies

Runs the full ablation suite to isolate the impact of hybrid semantic embeddings:
- **`DirectRL_Full`**: Uses the full RGAT + semantic embeddings pipeline.
- **`DirectRL_NoSemantic`**: Zeroes out indices `[15:783]` in `AblatedDataset` to analyze structural-only performance.
- **`ablate_structure`**: Restricts the graph to a shallow 1-layer message passing.

---

## 14. Repository Structure

```
SmellRL/
├── main.py                        # Pipeline entry point
├── config.yaml                    # All hyperparameters and paths
├── requirements.txt               # Python dependencies
├── Project_info.md                # This document (Technical Reference)
│
├── src/
│   ├── data.py        # MLCQ loading, AST parsing, graph builder (CONTAINS, CALLS, ACCESSES_FIELD)
│   ├── embeddings.py  # GraphCodeBERT cache-based embedder
│   ├── models.py      # NodeFeatureFusion, RGATLayer, RGATEncoder, SmellDetectionAgent
│   ├── training.py    # GCNPretrainer, DQNTrainer (with Soft Reward Matrix)
│   ├── evaluation.py  # Baselines mapping, ExperimentRunner, Refactoring Impact Simulator
│   └── utils.py       # CheckpointManager
```

---

*Document reflects SmellRL implementation as of July 2026.*
*Sources: src/data.py, src/models.py, src/training.py, src/evaluation.py, scripts/run_ablations.py, config.yaml*
