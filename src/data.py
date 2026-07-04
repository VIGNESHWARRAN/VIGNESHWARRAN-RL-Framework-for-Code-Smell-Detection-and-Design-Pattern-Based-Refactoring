"""
Data pipeline for SmellRL.

Stages:
  1. Load MLCQ CSV  (or generate synthetic data)
  2. Build a graph per instance (virtual graph from CK metrics, or AST if Java source is available)
  3. Serialize to torch_geometric Data objects
  4. Expose a PyTorch Dataset / DataLoader
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

from src.embeddings import SemanticEmbedder  # Added the embedder import

logger = logging.getLogger("SmellRL.data")

# ──────────────────────────── Constants ────────────────────────────────────

SMELL_CLASSES  = ["GodClass", "FeatureEnvy", "LongMethod", "DataClass", "NoSmell"]
PATTERN_CLASSES = ["Strategy", "Observer", "Facade", "Mediator", "ExtractClass", "None"]
SMELL_TO_IDX   = {s: i for i, s in enumerate(SMELL_CLASSES)}
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_CLASSES)}

SMELL_TO_PATTERNS: Dict[int, List[int]] = {
    0: [2, 3],   
    1: [0, 4],   
    2: [4, 5],   
    3: [1, 5],   
    4: [5],      
}

_CK_MAX = {"wmc": 150.0, "dit": 8.0, "noc": 20.0, "cbo": 40.0, "rfc": 200.0, "loc": 3000.0}

def _norm(val: float, key: str) -> float:
    return float(np.clip(val, 0, _CK_MAX[key]) / _CK_MAX[key])

# ──────────────────────────── MLCQ Loader ──────────────────────────────────

_SMELL_ALIASES = {
    "godclass": "GodClass", "god_class": "GodClass", "god class": "GodClass", "blob": "GodClass",
    "featureenvy": "FeatureEnvy", "feature_envy": "FeatureEnvy", "feature envy": "FeatureEnvy",
    "longmethod": "LongMethod", "long_method": "LongMethod", "long method": "LongMethod",
    "dataclass": "DataClass", "data_class": "DataClass", "data class": "DataClass",
    "nosmell": "NoSmell", "no_smell": "NoSmell", "no smell": "NoSmell",
    "none": "NoSmell",
}

def _normalize_smell(raw: str) -> Optional[str]:
    return _SMELL_ALIASES.get(str(raw).strip().lower())

def load_mlcq(csv_path: str, nosmell_ratio: float = 1.0) -> pd.DataFrame:
    logger.info(f"[Data] Loading MLCQ CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    smell_col = next(
        (c for c in ["smell", "smell_type", "kind", "codesmell", "code_smell", "smelltype", "type"] if c in df.columns),
        None,
    )
    if smell_col is None:
        raise ValueError(f"Cannot find smell-type column. Available: {list(df.columns)}")

    if "severity" in df.columns:
        df.loc[df["severity"].str.lower() == "none", smell_col] = "NoSmell"

    df["smell_label"] = df[smell_col].map(_normalize_smell)
    df["smell_idx"]   = df["smell_label"].map(SMELL_TO_IDX)

    df = df.dropna(subset=["smell_idx"]).reset_index(drop=True)
    df["smell_idx"] = df["smell_idx"].astype(int)

    df_smells = df[df["smell_label"] != "NoSmell"]
    df_nosmells = df[df["smell_label"] == "NoSmell"]

    if not df_nosmells.empty and not df_smells.empty:
        max_smell_count = df_smells["smell_label"].value_counts().max()
        # Downsample NoSmell to (nosmell_ratio * largest smell class) rather than a strict 1:1
        # match. A ratio of 1.0 reproduces the old behavior; >1.0 keeps more clean code so the
        # agent isn't trained to expect a smell in every 5th file, which was causing it to
        # over-flag healthy code at test time (hurting precision/F1).
        target_nosmell_count = int(max_smell_count * nosmell_ratio)
        if len(df_nosmells) > target_nosmell_count:
            df_nosmells_downsampled = df_nosmells.sample(n=target_nosmell_count, random_state=42)
            df = pd.concat([df_smells, df_nosmells_downsampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    ck_cols = ["wmc", "dit", "noc", "cbo", "rfc", "loc", "n_methods", "n_fields"]
    for col in ck_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in ck_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df

# ──────────────────────────── Graph Builder ────────────────────────────────

def _build_virtual_graph(row: pd.Series, embedder: SemanticEmbedder) -> Dict:
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

    # Base features
    class_feat = [
        1.0, 0.0, 0.0,
        _norm(wmc, "wmc"), _norm(dit, "dit"), _norm(noc, "noc"),
        _norm(cbo, "cbo"), _norm(rfc, "rfc"), _norm(loc, "loc"),
    ]
    
    # Actually embed the identifier using GraphCodeBERT
    identifier = str(row.get("code_name", row["smell_label"]))
    class_emb = embedder.embed_identifier(identifier).tolist()
    class_feat.extend(class_emb)

    # For virtual methods/fields without explicit names, use synthetic/zero embeddings to match dimensions
    zero_emb = [0.0] * embedder.dim
    
    method_feat = lambda: [
        0.0, 1.0, 0.0,
        min(1.0, avg_cc / 20.0), 0.0, 0.0, 0.0, 0.0,
        min(1.0, avg_loc / 300.0),
    ] + zero_emb
    
    field_feat = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + zero_emb

    node_features = [class_feat]
    node_features += [method_feat() for _ in range(n_m)]
    node_features += [field_feat for _ in range(n_f)]

    x = torch.tensor(node_features, dtype=torch.float32)

    src, dst = [], []
    for i in range(1, 1 + n_m + n_f):
        src.append(0); dst.append(i)   
        src.append(i); dst.append(0)   

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    y = torch.tensor(int(row["smell_idx"]), dtype=torch.long)
    return {"x": x, "edge_index": edge_index, "y": y}


def _build_ast_graph(row: pd.Series, source_code: str, embedder: SemanticEmbedder) -> Optional[Dict]:
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
        
        class_name = cls.name if hasattr(cls, 'name') else str(row.get("code_name", "Unknown"))
        class_feat.extend(embedder.embed_identifier(class_name).tolist())
        
        node_features = [class_feat]
        src, dst = [], []

        for i, m in enumerate(methods, start=1):
            stmts = sum(1 for _ in m.body) if m.body else 0
            cc = 1 + stmts * 0.3
            mloc = stmts * 1.5
            
            m_feat = [
                0.0, 1.0, 0.0,
                min(1.0, cc / 20.0), 0.0, 0.0, 0.0, 0.0,
                min(1.0, mloc / 200.0),
            ]
            m_name = m.name if hasattr(m, 'name') else "method"
            m_feat.extend(embedder.embed_identifier(m_name).tolist())
            node_features.append(m_feat)
            src += [0, i]; dst += [i, 0]

        for i, f in enumerate(fields, start=len(methods) + 1):
            f_name = f.declarators[0].name if hasattr(f, 'declarators') and f.declarators else "field"
            f_feat = [0.0, 0.0, 1.0] + [0.0] * 6
            f_feat.extend(embedder.embed_identifier(f_name).tolist())
            node_features.append(f_feat)
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
    def __init__(self, df: pd.DataFrame, embedder: SemanticEmbedder):
        self.df = df.reset_index(drop=True)
        self.embedder = embedder
        self._graphs: List[Dict] = []
        self._build_all_graphs()

    def _build_all_graphs(self):
        logger.info(f"[Data] Building graphs for {len(self.df)} instances... (This may take a moment due to embeddings)")
        n = len(self.df)
        log_every = max(1, n // 10)
        for i, row in self.df.iterrows():
            graph = None
            if "source_code" in self.df.columns and pd.notna(row.get("source_code")):
                graph = _build_ast_graph(row, str(row["source_code"]), self.embedder)
            if graph is None:
                graph = _build_virtual_graph(row, self.embedder)
            graph["idx"] = i
            self._graphs.append(graph)
            if (i + 1) % log_every == 0:
                logger.debug(f"[Data]   Built {i+1}/{n} graphs")
        logger.info(f"[Data] Graph construction complete.")

    def __len__(self): return len(self._graphs)
    def __getitem__(self, idx): return self._graphs[idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._graphs, path)
        logger.info(f"[Data] Saved {len(self._graphs)} graphs to {path}")

    @classmethod
    def load(cls, path: str) -> "SmellDataset":
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        obj._graphs = graphs
        return obj

# ──────────────────────────── Preprocessing Pipeline ───────────────────────

def run_preprocessing(cfg: dict) -> Tuple["SmellDataset", "SmellDataset", "SmellDataset"]:
    from sklearn.model_selection import train_test_split

    proc_dir  = cfg["paths"]["processed_dir"]
    train_path = os.path.join(proc_dir, "train.pt")
    val_path   = os.path.join(proc_dir, "val.pt")
    test_path  = os.path.join(proc_dir, "test.pt")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        logger.info("[Data] Processed .pt files found — loading cached graphs")
        return SmellDataset.load(train_path), SmellDataset.load(val_path), SmellDataset.load(test_path)

    df = load_mlcq(
        cfg["dataset"]["mlcq_csv"],
        nosmell_ratio=cfg["dataset"].get("nosmell_ratio", 1.0)
    )

    # Initialize the embedder once so it loads the model into memory
    embedder = SemanticEmbedder(cfg)

    seed = cfg["dataset"]["random_seed"]
    tr   = cfg["dataset"]["train_ratio"]
    vr   = cfg["dataset"]["val_ratio"]

    train_df, temp_df = train_test_split(df, test_size=1 - tr, stratify=df["smell_idx"], random_state=seed)
    val_size = vr / (1 - tr)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_size, stratify=temp_df["smell_idx"], random_state=seed)

    # Pass the embedder to actually extract the 768-dim features
    train_ds = SmellDataset(train_df, embedder)
    val_ds   = SmellDataset(val_df, embedder)
    test_ds  = SmellDataset(test_df, embedder)

    os.makedirs(proc_dir, exist_ok=True)
    train_ds.save(train_path)
    val_ds.save(val_path)
    test_ds.save(test_path)

    return train_ds, val_ds, test_ds

def collate_graphs(batch: List[Dict]) -> Dict:
    return batch