# SmellRL: Step-by-Step Data Flow

This document walks through exactly what happens to the data at every stage of the SmellRL pipeline, from raw MLCQ source code to a final refactored output with quality metrics. It is meant to be read alongside the two architecture diagrams already shared.

---

## Stage 0: Raw input — the MLCQ dataset

MLCQ is a dataset of real-world Java classes and methods, each labeled by human reviewers for the presence and severity of code smells (e.g. long method, god class, feature envy). At this stage the data is just:

- Raw source code (`.java` files or code snippets referenced by the CSV)
- A label per code unit: smell type + severity (none / minor / major / critical)

This is loaded by `src/data.py`. If no MLCQ CSV is present and `use_synthetic: true` is set in `config.yaml`, a synthetic dataset with the same schema is generated instead so the pipeline can still run end-to-end for testing.

**Output of this stage:** a table of (code unit, source text, smell label).

---

## Stage 1: Preprocessing — turning code into graphs

Neural networks can't read source code directly, so each code unit is converted into a **graph representation** that captures its structure.

1. The source code is parsed into an Abstract Syntax Tree (AST).
2. The AST is converted into a graph object: nodes represent code elements (methods, variables, control structures, class members), and edges represent relationships (calls, contains, inherits, data flow).
3. Each node is given an initial feature vector (e.g. token embeddings, node type, simple structural counts).
4. The resulting graph objects are serialized and cached to `data/processed/` so this expensive step doesn't need to be repeated on every run.

**Data transformation:** `source code (text)` → `AST` → `graph object (nodes, edges, node features)`.

This is run via `python main.py --stage preprocess` or automatically as the first step of the full pipeline.

---

## Stage 2: GCN pretraining — learning code embeddings

Before any reinforcement learning happens, a Graph Convolutional Network (GCN) is **supervised pretrained** to produce useful embeddings of code graphs.

1. Input: the graph objects from Stage 1, along with their ground-truth smell labels.
2. The GCN performs message passing — each node updates its representation based on its neighbors, repeated across several GCN layers. This lets local code structure (e.g. a deeply nested method calling many others) propagate into a single vector.
3. After message passing, the per-node embeddings are pooled (e.g. mean/max pooling) into a single fixed-length **graph embedding** representing the whole code unit.
4. This embedding is passed through a classification head and trained with a supervised loss against the smell labels, for `gcn_pretrain.epochs` (default 50) epochs.
5. The trained GCN weights are checkpointed to `checkpoints/gcn_pretrain_latest.pt`.

**Data transformation:** `graph object` → `node embeddings (via GCN layers)` → `pooled graph embedding (fixed-length vector)`.

**Why this matters for RL:** reinforcement learning is sample-inefficient and struggles to learn good *representations* and a *control policy* at the same time from scratch. By pretraining the GCN first, the RL agent starts with a graph embedding that already separates smelly from clean code structurally — RL then only has to learn the *decision policy* on top of that, not reinvent code understanding from raw graphs.

---

## Stage 3: Phase 1 — Smell detection via Deep Q-Network (DQN)

This is where reinforcement learning takes over.

### Setup
- **State:** the pooled graph embedding from the (pretrained, and optionally further fine-tuned) GCN encoder for a given code unit. `training.freeze_gcn` controls whether the GCN weights keep updating during RL or stay fixed.
- **Action space:** one action per smell category (including "no smell"), i.e. the agent must classify the code unit's smell type.
- **Q-network:** a small feed-forward network on top of the graph embedding that outputs a Q-value (expected future reward) for each possible smell label.

### The training loop (repeated every episode/step)
1. **Observe state** — encode the current code unit into its graph embedding.
2. **Select action** — with probability ε pick a random action (exploration), otherwise pick the action with the highest predicted Q-value (exploitation). ε decays over training so the agent explores early and exploits later.
3. **Receive reward** — compare the predicted smell label to the ground-truth MLCQ label. Correct classification gives positive reward; incorrect gives negative or zero reward, often scaled by severity (misclassifying a critical smell is penalized more than a minor one).
4. **Store transition** — the tuple `(state, action, reward, next_state)` is pushed into a replay buffer.
5. **Sample and update** — periodically, a random mini-batch of past transitions is sampled from the replay buffer (this breaks correlation between consecutive samples) and used to update the Q-network via gradient descent on the temporal-difference (TD) error.
6. This repeats for `training.max_episodes` (default 500) episodes, with a checkpoint saved after each one to `checkpoints/smellrl_latest.pt` so training can resume if interrupted.

**Why RL instead of plain supervised classification here?** Framing detection as a sequential decision problem lets the same architecture later be extended to multi-step refactoring decisions (Phase 2), and lets the reward function encode severity-aware, non-uniform costs that a standard cross-entropy loss doesn't naturally express.

**Output of Phase 1:** a trained agent that, given any code unit's embedding, predicts which smell (if any) it has — this is what produces `exp1_phase1_smell_detection.csv`.

---

## Stage 4: Phase 2 — Design pattern recommendation via DQN

Once a code unit has been flagged with a smell in Phase 1, Phase 2 decides *how to fix it* by recommending a design pattern (e.g. Extract Method, Strategy, Factory) appropriate for that smell.

### Setup
- **State:** the same graph embedding, now optionally concatenated with or conditioned on the Phase 1 output (the detected smell type), since the right pattern depends on which smell is present.
- **Action space:** one action per candidate design pattern in the refactoring catalog.
- **Q-network:** structurally similar to Phase 1's, but with a different output head sized to the pattern catalog.

### The training loop
This follows the exact same state → action → reward → replay → update cycle described in Stage 3, with one key difference in the **reward signal**: instead of (or in addition to) label-matching reward, Phase 2's reward is computed from **code quality metrics before and after the recommended pattern is conceptually applied**:

- **ΔMI** — change in Maintainability Index
- **ΔCBO** — change in Coupling Between Objects
- **ΔCC** — change in Cyclomatic Complexity

A pattern recommendation that improves these metrics (higher MI, lower CBO/CC) is rewarded; one that makes them worse is penalized. This ties the RL objective directly to measurable software engineering quality, rather than just matching a human-labeled "correct" pattern.

**Output of Phase 2:** a trained agent producing pattern recommendations, captured in `exp2_phase2_pattern.csv`, with the underlying metric deltas recorded in `exp3_quality_impact.json`.

---

## Stage 5: Baselines and ablations

To contextualize SmellRL's performance, the same pipeline supports training comparison models:

- **Random agent** — picks actions uniformly at random; establishes the performance floor.
- **Rule-based** — hand-coded heuristics (e.g. thresholding method length, cyclomatic complexity) without learning.
- **Flat DQN** (`python main.py --stage train_flat`) — a single-phase DQN trained directly on raw graph features without the hierarchical Phase 1 → Phase 2 split and without GCN pretraining, used to isolate how much the hierarchical, pretrained design contributes.
- **Ablations** (`python scripts/run_ablations.py`) — systematically removes/disables individual components (e.g. freezing the GCN, removing the replay buffer, disabling severity-weighted reward) to measure each component's contribution, saved to `ablations.csv`.

---

## Stage 6: Evaluation and final output

`src/evaluation.py` runs all 5 experiments (or `python main.py --stage experiment` once both phases are trained) and produces:

| File | What it captures |
|---|---|
| `exp1_phase1_smell_detection.csv` | Phase-1 detection accuracy/F1 vs. Random, Rule-Based, Flat DQN baselines |
| `exp2_phase2_pattern.csv` | Phase-2 pattern recommendation accuracy |
| `exp3_quality_impact.json` | Measured ΔMI, ΔCBO, ΔCC from applying recommended patterns |
| `exp4_smellrl_curve.csv` | Reward/accuracy per training episode (learning curve) |
| `exp5_per_class_f1.csv` | F1 broken down per individual smell category |
| `ablations.csv` | Performance with each architectural component removed |

The final practical output is, for a given input codebase: each code unit's predicted smell label (Phase 1) plus, where a smell was detected, a recommended design pattern (Phase 2) and the expected quality improvement (ΔMI, ΔCBO, ΔCC) that pattern would yield — i.e. an actionable, prioritized refactoring report rather than just a flat list of warnings.

---

## End-to-end data shape summary

| Stage | Input shape | Output shape |
|---|---|---|
| Raw data | source code text | (code unit, label) pairs |
| Preprocessing | source code | graph object (nodes, edges, features) |
| GCN pretraining | graph object | pooled graph embedding (fixed-length vector) |
| Phase 1 RL | graph embedding | smell label + Q-values per smell class |
| Phase 2 RL | graph embedding (+ smell label) | recommended pattern + Q-values per pattern |
| Evaluation | trained agents + held-out data | accuracy/F1 metrics, ΔMI/ΔCBO/ΔCC, learning curves |

Throughout, every stage's progress is checkpointed (`gcn_pretrain_latest.pt`, `smellrl_latest.pt`, `flat_dqn_latest.pt`), so `python main.py` can always be re-run to resume from the last completed episode/epoch rather than restarting the whole pipeline.
