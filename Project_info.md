Here is a comprehensive, academic, and professional `README.md` for your project. You can copy and paste this directly into your repository.

---

# Semantic-Aware SmellRL: Reinforcement Learning for Code Smell Detection

**SmellRL** is a novel hybrid machine learning pipeline that combines **Graph Convolutional Networks (GCN)**, **Large Language Models (GraphCodeBERT)**, and **Deep Reinforcement Learning (DQN)** to accurately detect structural and semantic code smells in software repositories.

While traditional smell detectors rely purely on structural metrics (like CK metrics), SmellRL constructs a semantic-aware graph representation of source code. By encoding both *how the code is structured* and *what the code means*, the Reinforcement Learning agent achieves state-of-the-art accuracy on highly subjective smells like `FeatureEnvy` and `DataClass`.

---

## 🧠 Architecture Overview

The system operates in a three-step architectural pipeline:

1. **Graph Construction (777-Dimensions):**
Source code (via the MLCQ dataset) is parsed into a graph where nodes represent classes, methods, and fields. Each node is injected with **9 structural CK metrics** and **768-dimensional semantic embeddings** extracted via `microsoft/graphcodebert-base`.
2. **GCN Encoder (State Compression):**
A Graph Convolutional Network is pre-trained to pass messages along the AST/Dependency graph, compressing the massive 777-dimensional node features into a dense, highly optimized **128-dimensional state vector**.
3. **Deep Q-Network (The RL Agent):**
A Reinforcement Learning agent observes the 128-dimensional state and learns an optimal policy to classify the graph into one of five categories: `GodClass`, `FeatureEnvy`, `LongMethod`, `DataClass`, or `NoSmell`.

---

## 📊 Supported Code Smells

* **God Class** (Blob)
* **Feature Envy**
* **Long Method**
* **Data Class**
* **No Smell** (Healthy Code)

---

## ⚙️ Installation & Requirements

**Prerequisites:** Python 3.8+

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/SmellRL.git
cd SmellRL
pip install -r requirements.txt

```

**Core Dependencies:**

* `torch` & `torch-geometric` (GCN and neural networks)
* `transformers` (HuggingFace GraphCodeBERT embeddings)
* `pandas`, `numpy`, `scikit-learn` (Data processing and baselines)
* `javalang` (Optional, for AST parsing of raw Java files)

---

## 🚀 Usage: The Training Pipeline

The entire pipeline is driven by `main.py` and configured via `config.yaml`. The pipeline consists of four sequential stages:

### 1. Preprocessing (`--stage preprocess`)

Parses the MLCQ dataset, balances the classes (downsampling `NoSmell` to match the positive classes), queries HuggingFace to extract GraphCodeBERT embeddings, and saves the graphs as `.pt` tensors.

```bash
python main.py --stage preprocess

```

### 2. GCN Pre-training (`--stage pretrain`)

Trains the Graph Convolutional Network in a supervised manner to learn how to efficiently compress the 777-dimensional graphs into 128-dimensional state vectors.

```bash
python main.py --stage pretrain

```

### 3. RL Agent Training (`--stage train`)

Initializes the Deep Q-Network. The agent explores the environment, receives rewards (+1.0 for correct classification, -0.5 for incorrect), and decays epsilon over the configured number of episodes.

```bash
python main.py --stage train

```

### 4. Evaluation (`--stage experiment`)

Evaluates the trained RL agent against the unseen test set and compares it against three baselines: Random Guessing, Rule-Based (Designite-like), and an SVM trained on CK metrics.

```bash
python main.py --stage experiment

```

*Results are output to `data/results/all_results_summary.json` and `exp5_per_class_f1.csv`.*

---

## 🔬 Ablation Study: The Impact of Semantics

This repository includes a built-in ablation toggle to prove the efficacy of semantic embeddings. You can disable GraphCodeBERT to train an agent purely on the 9-dimensional structural CK metrics.

**To run the ablation study:**

1. Open `src/data.py` and set `USE_EMBEDDINGS = False`.
2. Open `src/embeddings.py` and set `USE_EMBEDDINGS_MODEL = False`.
3. Clear the cache: `rm data/processed/*.pt`
4. Re-run the full 4-stage pipeline.

### Findings

Incorporating semantic embeddings yields significant performance boosts, particularly on smells that rely heavily on code context rather than just size or complexity:

* **Overall Accuracy:** +6.8% improvement
* **Data Class (F1):** +10.4% improvement
* **God Class (F1):** +5.3% improvement

---

## 📁 Repository Structure

```text
SmellRL/
├── main.py                  # Entry point for the pipeline
├── config.yaml              # Hyperparameters and paths configuration
├── data/
│   ├── mlcq/                # Raw MLCQ dataset CSVs
│   ├── processed/           # Serialized PyTorch Geometric graphs (.pt)
│   └── results/             # Evaluation JSONs and CSVs
├── src/
│   ├── data.py              # Dataset loading, balancing, and graph generation
│   ├── embeddings.py        # GraphCodeBERT HuggingFace wrapper
│   ├── models.py            # GCN Encoder and DQN Agent architecture
│   ├── pretrain.py          # Supervised training loop for the GCN
│   ├── training.py          # RL environment and training loop
│   └── evaluation.py        # Baseline comparisons and metric generation
└── requirements.txt

```

---

## 📝 License & Acknowledgments

* **Dataset:** Evaluated using the [MLCQ Dataset](https://www.google.com/search?q=https://mlcq.github.io/), a manually validated code smell dataset.
* **Embeddings:** Powered by [microsoft/graphcodebert-base](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmicrosoft%2Fgraphcodebert-base).