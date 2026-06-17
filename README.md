# SmellRL

A Two-Phase Hierarchical DQN framework for Code Smell Detection and Design Pattern-Based Refactoring, trained on the MLCQ dataset.

## Quick Start (GPU laptop)

```bash
# 1. Clone and enter the repo
git clone <your-github-url>
cd SmellRL

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install PyTorch with CUDA (check your CUDA version first with: nvidia-smi)
#    For CUDA 12.x:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#    For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Get the MLCQ dataset (two options)
#    Option A — auto-download from Zenodo:
python scripts/download_mlcq.py
#    Option B — manual: place your mlcq.csv at data/mlcq/mlcq.csv
#    Option C — use synthetic data (no download needed):
#      Edit config.yaml and set  use_synthetic: true

# 6. Run the full pipeline (auto-resumes from checkpoint if interrupted)
python main.py
```

## Usage

```bash
# Run individual stages
python main.py --stage preprocess      # build graph datasets only
python main.py --stage pretrain        # GCN supervised pre-training
python main.py --stage train           # SmellRL RL training
python main.py --stage train_flat      # Flat DQN baseline training
python main.py --stage experiment      # run all 5 experiments (needs trained models)

# Run ablation studies (Section 7)
python scripts/run_ablations.py

# Resume after interruption — just re-run the same command
python main.py                         # picks up from latest checkpoint automatically
```

## Project Structure

```
SmellRL/
├── config.yaml               # all hyperparameters
├── main.py                   # single entry point
├── requirements.txt
├── src/
│   ├── data.py               # MLCQ loading, graph building, dataset
│   ├── models.py             # GCN encoder, Q-networks, RL agent
│   ├── training.py           # replay buffer, reward, training loops
│   ├── evaluation.py         # metrics, baselines, experiment runner
│   └── utils.py              # logger, checkpoint manager
├── scripts/
│   ├── download_mlcq.py      # fetch MLCQ from Zenodo
│   └── run_ablations.py      # ablation study runner
├── data/
│   ├── mlcq/                 # raw MLCQ CSV (gitignored)
│   ├── processed/            # serialised graph objects (gitignored)
│   └── results/              # experiment CSVs and JSON (committed)
├── checkpoints/              # model checkpoints (gitignored)
└── logs/                     # training logs (gitignored)
```

## Resume After Interruption

Every stage saves a `<stage>_latest.pt` checkpoint after each episode/epoch.
Simply re-run the same command — it will load the latest checkpoint and continue.

Checkpoints stored at `checkpoints/`:
- `gcn_pretrain_latest.pt`
- `smellrl_latest.pt`
- `flat_dqn_latest.pt`

## Experiments

All 5 experiments run automatically via `python main.py` or `python main.py --stage experiment`.
Results are saved to `data/results/`:

| File | Experiment |
|------|------------|
| `exp1_phase1_smell_detection.csv` | Phase-1 smell detection vs baselines |
| `exp2_phase2_pattern.csv` | Phase-2 pattern recommendation |
| `exp3_quality_impact.json` | ΔMI, ΔCBO, ΔCC after pattern application |
| `exp4_smellrl_curve.csv` | Learning curve per episode |
| `exp5_per_class_f1.csv` | Per-class F1 breakdown |
| `ablations.csv` | Ablation study results |

## Configuration

Key settings in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `device` | `auto` | `auto` detects CUDA, else CPU |
| `training.max_episodes` | `500` | number of training episodes |
| `training.freeze_gcn` | `false` | freeze GCN during RL training |
| `dataset.use_synthetic` | `false` | use synthetic data if no CSV |
| `gcn_pretrain.epochs` | `50` | GCN pre-training epochs |

## Expected Results (from paper)

| Method | Phase-1 F1 | Joint Accuracy |
|--------|-----------|----------------|
| Random Agent | ~0.19 | ~0.04 |
| Rule-Based | ~0.70 | ~0.49 |
| Flat DQN | ~0.73 | ~0.53 |
| **SmellRL (ours)** | **~0.82** | **~0.70** |
