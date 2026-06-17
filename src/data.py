"""
Data pipeline for SmellRL.

Stages:
  1. Load MLCQ CSV  (or generate synthetic data)
  2. Build a graph per instance (virtual graph from CK metrics, or AST if Java source is available)
  3. Serialize to torch_geometric Data objects
  4. Expose a PyTorch Dataset / DataLoader

Graph schema (per instance)
  Nodes:
    - index 0: the class node  → features [1,0,0, wmc_n, dit_n, noc_n, cbo_n, rfc_n, loc_n]
    - index 1..M: method nodes → features [0,1,0, avg_cc_n, 0, 0, 0, 0, avg_mloc_n]
    - index M+1..M+F: field nodes → features [0,0,1, 0,…,0]
  Edges: class→method (contains), class→field (contains)
  Node feature dim = 9

If Java source is provided alongside the CSV (column 'source_code'), a richer AST
graph is built using the javalang parser.  Otherwise the virtual graph is used.
"""

import os
import logging
import math
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("SmellRL.data")

# ──────────────────────────── Constants ────────────────────────────────────

SMELL_CLASSES  = ["GodClass", "FeatureEnvy", "LongMethod", "DataClass", "NoSmell"]
PATTERN_CLASSES = ["Strategy", "Observer", "Facade", "Mediator", "ExtractClass", "None"]
SMELL_TO_IDX   = {s: i for i, s in enumerate(SMELL_CLASSES)}
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_CLASSES)}

# Ground-truth mapping: smell_idx → list of valid pattern_idx
SMELL_TO_PATTERNS: Dict[int, List[int]] = {
    0: [2, 3],   # GodClass    → Facade, Mediator
    1: [0, 4],   # FeatureEnvy → Strategy, ExtractClass
    2: [4, 5],   # LongMethod  → ExtractClass, None
    3: [1, 5],   # DataClass   → Observer, None
    4: [5],      # NoSmell     → None
}

NODE_FEATURE_DIM = 9

# CK metric normalisation constants (95th-percentile reference values)
_CK_MAX = {"wmc": 150.0, "dit": 8.0, "noc": 20.0, "cbo": 40.0, "rfc": 200.0, "loc": 3000.0}


def _norm(val: float, key: str) -> float:
    return float(np.clip(val, 0, _CK_MAX[key]) / _CK_MAX[key])


# ──────────────────────────── MLCQ Loader ──────────────────────────────────

_SMELL_ALIASES = {
    "godclass": "GodClass", "god_class": "GodClass", "god class": "GodClass",
    "featureenvy": "FeatureEnvy", "feature_envy": "FeatureEnvy", "feature envy": "FeatureEnvy",
    "longmethod": "LongMethod", "long_method": "LongMethod", "long method": "LongMethod",
    "dataclass": "DataClass", "data_class": "DataClass", "data class": "DataClass",
    "nosmell": "NoSmell", "no_smell": "NoSmell", "no smell": "NoSmell",
    "none": "NoSmell",
}


def _normalize_smell(raw: str) -> Optional[str]:
    return _SMELL_ALIASES.get(str(raw).strip().lower())


def load_mlcq(csv_path: str) -> pd.DataFrame:
    """
    Load MLCQ CSV.  Handles several common column layouts:
      - (smell_type / kind / type) + optional severity column
      - Optional pre-computed CK metric columns
      - Optional source_code column
    Returns a normalised DataFrame with columns:
      smell_label, smell_idx, wmc, dit, noc, cbo, rfc, loc, n_methods, n_fields
    """
    logger.info(f"[Data] Loading MLCQ CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    logger.info(f"[Data] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # ── Find smell column ──────────────────────────────────────────────────
    smell_col = next(
        (c for c in ["smell_type", "kind", "type", "codesmell", "code_smell", "smelltype"] if c in df.columns),
        None,
    )
    if smell_col is None:
        raise ValueError(f"Cannot find smell-type column. Available: {list(df.columns)}")

    # severity=none → NoSmell
    if "severity" in df.columns:
        df.loc[df["severity"].str.lower() == "none", smell_col] = "NoSmell"

    df["smell_label"] = df[smell_col].map(_normalize_smell)
    df["smell_idx"]   = df["smell_label"].map(SMELL_TO_IDX)

    before = len(df)
    df = df.dropna(subset=["smell_idx"]).reset_index(drop=True)
    df["smell_idx"] = df["smell_idx"].astype(int)
    if before != len(df):
        logger.warning(f"[Data] Dropped {before - len(df)} rows with unknown smell labels")

    # ── CK metrics (use if present, else fill zeros) ───────────────────────
    ck_cols = ["wmc", "dit", "noc", "cbo", "rfc", "loc", "n_methods", "n_fields"]
    for col in ck_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in ck_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    logger.info(f"[Data] Smell distribution:\n{df['smell_label'].value_counts().to_string()}")
    return df


# ──────────────────────────── Synthetic Data ───────────────────────────────

_SMELL_METRIC_PROFILES = {
    "GodClass":    dict(wmc=(52,15), dit=(2,1), noc=(1,1),  cbo=(12,4), rfc=(65,20), loc=(850,200), n_methods=(22,8),  n_fields=(14,5),  avg_cc=(4.5,1.5)),
    "FeatureEnvy": dict(wmc=(20,8),  dit=(2,1), noc=(1,1),  cbo=(9,3),  rfc=(40,12), loc=(250,80),  n_methods=(8,3),   n_fields=(5,3),   avg_cc=(3.5,1.0)),
    "LongMethod":  dict(wmc=(30,10), dit=(2,1), noc=(1,1),  cbo=(7,2),  rfc=(35,10), loc=(400,120), n_methods=(7,3),   n_fields=(5,2),   avg_cc=(6.0,2.0)),
    "DataClass":   dict(wmc=(8,3),   dit=(1,1), noc=(0,0.5),cbo=(4,2),  rfc=(15,5),  loc=(150,50),  n_methods=(4,2),   n_fields=(10,4),  avg_cc=(1.5,0.5)),
    "NoSmell":     dict(wmc=(15,7),  dit=(2,1), noc=(1,0.5),cbo=(5,2),  rfc=(25,8),  loc=(200,80),  n_methods=(5,2),   n_fields=(4,2),   avg_cc=(2.5,0.8)),
}
_SMELL_COUNTS = {"GodClass": 921, "FeatureEnvy": 843, "LongMethod": 1104, "DataClass": 687, "NoSmell": 522}


def generate_synthetic(n: int = 4077, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic MLCQ-like instances (used when CSV is unavailable)."""
    np.random.seed(seed)
    logger.info(f"[Data] Generating {n} synthetic instances (seed={seed})")

    total = sum(_SMELL_COUNTS.values())
    smells: List[str] = []
    for s, cnt in _SMELL_COUNTS.items():
        smells.extend([s] * round(n * cnt / total))
    while len(smells) < n:
        smells.append("LongMethod")
    smells = smells[:n]
    np.random.shuffle(smells)

    rows = []
    for smell in smells:
        p = _SMELL_METRIC_PROFILES[smell]
        r = {
            "smell_label": smell,
            "smell_idx":   SMELL_TO_IDX[smell],
        }
        for key, (mu, sigma) in p.items():
            r[key] = max(0.0, np.random.normal(mu, sigma))
        rows.append(r)

    df = pd.DataFrame(rows)
    # derive n_methods from wmc if not set
    if "n_methods" not in df.columns:
        df["n_methods"] = (df["wmc"] / 3.0).clip(1).round().astype(int)
    if "n_fields" not in df.columns:
        df["n_fields"] = (df["loc"] / 30.0).clip(1).round().astype(int)
    logger.info(f"[Data] Synthetic distribution:\n{df['smell_label'].value_counts().to_string()}")
    return df


# ──────────────────────────── Graph Builder ────────────────────────────────

try:
    from torch_geometric.data import Data as PyGData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("[Data] torch_geometric not found — graphs stored as dict tensors")


def _build_virtual_graph(row: pd.Series) -> Dict:
    """
    Build a graph from CK metrics without Java source.
    Returns a dict with keys: x, edge_index, y (all torch.Tensor).
    """
    wmc = float(row.get("wmc", 0))
    dit = float(row.get("dit", 0))
    noc = float(row.get("noc", 0))
    cbo = float(row.get("cbo", 0))
    rfc = float(row.get("rfc", 0))
    loc = float(row.get("loc", 0))
    n_m = max(1, int(row.get("n_methods", max(1, wmc / 3))))
    n_f = max(0, int(row.get("n_fields", max(0, loc / 30))))
    avg_cc  = float(row.get("avg_cc", wmc / max(1, n_m)))
    avg_loc = loc / max(1, n_m)

    # ── Node features ─────────────────────────────────────────────────────
    class_feat = [
        1.0, 0.0, 0.0,
        _norm(wmc, "wmc"), _norm(dit, "dit"), _norm(noc, "noc"),
        _norm(cbo, "cbo"), _norm(rfc, "rfc"), _norm(loc, "loc"),
    ]
    method_feat = lambda: [
        0.0, 1.0, 0.0,
        min(1.0, avg_cc / 20.0), 0.0, 0.0, 0.0, 0.0,
        min(1.0, avg_loc / 300.0),
    ]
    field_feat = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    node_features = [class_feat]
    node_features += [method_feat() for _ in range(n_m)]
    node_features += [field_feat for _ in range(n_f)]

    x = torch.tensor(node_features, dtype=torch.float32)

    # ── Edges (class → method, class → field) ─────────────────────────────
    src, dst = [], []
    for i in range(1, 1 + n_m + n_f):
        src.append(0); dst.append(i)   # class → child
        src.append(i); dst.append(0)   # child → class (bidirectional)

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor(int(row["smell_idx"]), dtype=torch.long)
    return {"x": x, "edge_index": edge_index, "y": y}


def _build_ast_graph(row: pd.Series, source_code: str) -> Optional[Dict]:
    """
    Parse Java source with javalang and build a richer AST graph.
    Falls back to virtual graph on parse errors.
    """
    try:
        import javalang

        tree = javalang.parse.parse(source_code)
        if not tree.types:
            return None

        cls = tree.types[0]
        methods = list(cls.methods) if hasattr(cls, "methods") else []
        fields  = list(cls.fields)  if hasattr(cls, "fields")  else []

        wmc = float(row.get("wmc", len(methods) * 3))
        dit = float(row.get("dit", 0))
        noc = float(row.get("noc", 0))
        cbo = float(row.get("cbo", 0))
        rfc = float(row.get("rfc", 0))
        loc = float(row.get("loc", len(source_code.splitlines())))

        class_feat = [
            1.0, 0.0, 0.0,
            _norm(wmc, "wmc"), _norm(dit, "dit"), _norm(noc, "noc"),
            _norm(cbo, "cbo"), _norm(rfc, "rfc"), _norm(loc, "loc"),
        ]

        node_features = [class_feat]
        src, dst = [], []

        for i, m in enumerate(methods, start=1):
            # Estimate CC from number of statements (proxy)
            stmts = sum(1 for _ in m.body) if m.body else 0
            cc = 1 + stmts * 0.3
            params = len(m.parameters) if m.parameters else 0
            mloc = stmts * 1.5
            node_features.append([
                0.0, 1.0, 0.0,
                min(1.0, cc / 20.0), 0.0, 0.0, 0.0, 0.0,
                min(1.0, mloc / 200.0),
            ])
            src += [0, i]; dst += [i, 0]

        for i, _ in enumerate(fields, start=len(methods) + 1):
            node_features.append([0.0, 0.0, 1.0] + [0.0] * 6)
            src += [0, i]; dst += [i, 0]

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        y = torch.tensor(int(row["smell_idx"]), dtype=torch.long)
        return {"x": x, "edge_index": edge_index, "y": y}

    except Exception as e:
        logger.debug(f"[Data] AST parse failed ({e}), using virtual graph")
        return None


# ──────────────────────────── Dataset ──────────────────────────────────────

class SmellDataset(Dataset):
    """
    PyTorch Dataset over MLCQ instances.

    Each item is a dict:
      x          : node features  [N_nodes × 9]
      edge_index : [2 × N_edges]
      y          : smell label (long scalar)
      idx        : position in dataset
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._graphs: List[Dict] = []
        self._build_all_graphs()

    def _build_all_graphs(self):
        logger.info(f"[Data] Building graphs for {len(self.df)} instances...")
        n = len(self.df)
        log_every = max(1, n // 10)
        for i, row in self.df.iterrows():
            graph = None
            if "source_code" in self.df.columns and pd.notna(row.get("source_code")):
                graph = _build_ast_graph(row, str(row["source_code"]))
            if graph is None:
                graph = _build_virtual_graph(row)
            graph["idx"] = i
            self._graphs.append(graph)
            if (i + 1) % log_every == 0:
                logger.debug(f"[Data]   Built {i+1}/{n} graphs")
        logger.info(f"[Data] Graph construction complete.")

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._graphs, path)
        logger.info(f"[Data] Saved {len(self._graphs)} graphs to {path}")

    @classmethod
    def load(cls, path: str) -> "SmellDataset":
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        obj._graphs = graphs
        logger.info(f"[Data] Loaded {len(graphs)} graphs from {path}")
        return obj


# ──────────────────────────── Preprocessing Pipeline ───────────────────────

def run_preprocessing(cfg: dict) -> Tuple["SmellDataset", "SmellDataset", "SmellDataset"]:
    """
    Full pipeline: CSV → split DataFrames → graph Datasets → saved .pt files.
    If .pt files already exist, loads them directly (resume support).
    Returns (train_ds, val_ds, test_ds).
    """
    from sklearn.model_selection import train_test_split

    proc_dir  = cfg["paths"]["processed_dir"]
    train_path = os.path.join(proc_dir, "train.pt")
    val_path   = os.path.join(proc_dir, "val.pt")
    test_path  = os.path.join(proc_dir, "test.pt")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        logger.info("[Data] Processed .pt files found — loading cached graphs")
        return SmellDataset.load(train_path), SmellDataset.load(val_path), SmellDataset.load(test_path)

    # Load raw data
    use_synthetic = cfg["dataset"].get("use_synthetic", False)
    csv_path = cfg["dataset"]["mlcq_csv"]

    if use_synthetic or not os.path.exists(csv_path):
        if not use_synthetic:
            logger.warning(f"[Data] MLCQ CSV not found at {csv_path}. Using synthetic data.")
        df = generate_synthetic(
            n=cfg["dataset"].get("n_synthetic", 4077),
            seed=cfg["dataset"]["random_seed"],
        )
    else:
        df = load_mlcq(csv_path)

    # Stratified split
    seed = cfg["dataset"]["random_seed"]
    tr   = cfg["dataset"]["train_ratio"]
    vr   = cfg["dataset"]["val_ratio"]

    train_df, temp_df = train_test_split(
        df, test_size=1 - tr, stratify=df["smell_idx"], random_state=seed
    )
    val_size = vr / (1 - tr)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_size, stratify=temp_df["smell_idx"], random_state=seed
    )
    logger.info(f"[Data] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_ds = SmellDataset(train_df)
    val_ds   = SmellDataset(val_df)
    test_ds  = SmellDataset(test_df)

    os.makedirs(proc_dir, exist_ok=True)
    train_ds.save(train_path)
    val_ds.save(val_path)
    test_ds.save(test_path)

    return train_ds, val_ds, test_ds


# ──────────────────────────── Collate / Batch ──────────────────────────────

def collate_graphs(batch: List[Dict]) -> Dict:
    """
    Simple collate: returns the list as-is.
    The trainer processes one instance at a time for the RL loop.
    """
    return batch
