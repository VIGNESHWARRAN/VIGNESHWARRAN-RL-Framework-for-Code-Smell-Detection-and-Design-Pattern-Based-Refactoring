import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import javalang

from src.embeddings import SemanticEmbedder

logger = logging.getLogger("SmellRL.data")

EDGE_TYPE = {
    "CONTAINS":       0,
    "CALLS":          1,
    "ACCESSES_FIELD": 2,
}
NUM_EDGE_TYPES = 3

DATA_VERSION = "v3"

SMELL_CLASSES  = ["GodClass", "FeatureEnvy", "LongMethod", "DataClass", "NoSmell"]
PATTERN_CLASSES = ["Strategy", "Observer", "Facade", "Mediator", "ExtractClass", "None"]
SMELL_TO_IDX   = {s: i for i, s in enumerate(SMELL_CLASSES)}
PATTERN_TO_IDX = {p: i for i, p in enumerate(PATTERN_CLASSES)}

_HALSTEAD_MAX = {
    "cyclomatic": 50.0,
    "lloc": 5000.0,
    "V": 5000.0,
    "D": 100.0,
    "E": 1000000.0,
    "B": 5.0,
    "dit": 1.0,
    "noc": 5.0,
    "cbo": 100.0,
    "rfc": 5000.0,
    "loc": 5000.0,
    "wmc": 50.0,
}

def _norm(val: float, key: str) -> float:
    return float(np.clip(val, 0, _HALSTEAD_MAX.get(key, 100.0)) / _HALSTEAD_MAX.get(key, 100.0))

# ──────────────────────────── AST Helpers ──────────────────────────────────

def compute_cc(method_node) -> int:
    cc = 1
    if not hasattr(method_node, 'body') or method_node.body is None:
        return cc
    for _, node in method_node.filter(javalang.tree.IfStatement): cc += 1
    for _, node in method_node.filter(javalang.tree.WhileStatement): cc += 1
    for _, node in method_node.filter(javalang.tree.ForStatement): cc += 1
    for _, node in method_node.filter(javalang.tree.SwitchStatementCase): cc += 1
    for _, node in method_node.filter(javalang.tree.CatchClause): cc += 1
    return cc

def extract_node_code(node, source_lines: List[str], next_line_num: Optional[int]) -> str:
    if not hasattr(node, 'position') or not node.position:
        return getattr(node, 'name', 'unknown')
    start = max(0, node.position.line - 1)
    end = next_line_num - 1 if next_line_num else len(source_lines)
    end = min(end, start + 50)
    return "\n".join(source_lines[start:end])

# ─────────────────────────── SmellyCode++ Loader ────────────────────────────

SMELL_PRIORITY = ["GodClass", "FeatureEnvy", "LongMethod", "DataClass"]

def dominant_smell(row: pd.Series) -> str:
    for smell in SMELL_PRIORITY:
        if row.get(smell, 0) == 1:
            return smell
    return "NoSmell"

def load_smellycode(csv_path: str, nosmell_ratio: float = 3.0) -> pd.DataFrame:
    logger.info(f"[Data] Loading SmellyCode++ CSV from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SmellyCode++ CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    REQUIRED_COLS = {"Code", "GodClass", "FeatureEnvy", "LongMethod", "DataClass",
                     "lloc", "cyclomatic", "V", "D", "E", "B"}
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"SmellyCode++ CSV missing columns: {missing}. Got: {list(df.columns)}")

    logger.info(f"[Data] SmellyCode++ loaded: {len(df)} raw rows")

    # Map multi-label to dominant smell
    df["smell_label"] = df.apply(dominant_smell, axis=1)
    df["smell_idx"]   = df["smell_label"].map(SMELL_TO_IDX)
    df = df.dropna(subset=["smell_idx"]).reset_index(drop=True)
    df["smell_idx"]   = df["smell_idx"].astype(int)

    # Rename Code to source_code
    df = df.rename(columns={"Code": "source_code"})

    # Class balancing
    df_smells = df[df["smell_label"] != "NoSmell"]
    df_nsmell = df[df["smell_label"] == "NoSmell"]

    logger.info("[Data] Class distribution before downsampling:")
    for smell_cls in SMELL_CLASSES:
        count = len(df[df["smell_label"] == smell_cls])
        logger.info(f"  {smell_cls}: {count}")

    if not df_nsmell.empty and not df_smells.empty:
        max_smell_count = df_smells["smell_label"].value_counts().max()
        target_nosmell_count = int(max_smell_count * nosmell_ratio)
        if len(df_nsmell) > target_nosmell_count:
            df_nsmell_downsampled = df_nsmell.sample(n=target_nosmell_count, random_state=42)
            df = pd.concat([df_smells, df_nsmell_downsampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info("[Data] Class distribution after downsampling:")
    for smell_cls in SMELL_CLASSES:
        count = len(df[df["smell_label"] == smell_cls])
        logger.info(f"  {smell_cls}: {count}")

    return df

# ──────────────────────────── Graph Builder ────────────────────────────────

def _build_ast_graph(row: pd.Series, source_code: str, embedder: SemanticEmbedder) -> Dict:
    # No silent try-except fallback that returns None or empty graph
    try:
        try:
            tree = javalang.parse.parse(source_code)
        except Exception:
            tree = javalang.parse.parse(f"class _Dummy {{ {source_code} }}")
    except Exception as e:
        logger.error(f"[Data] Failed parsing AST for row: {e}")
        raise ValueError(f"AST parsing failed for code snippet: {source_code[:200]}") from e

    if not tree.types:
        raise ValueError(f"AST parsing produced empty types list for code snippet: {source_code[:200]}")

    cls = tree.types[0]
    methods = list(cls.methods) if hasattr(cls, "methods") else []
    fields  = list(cls.fields)  if hasattr(cls, "fields")  else []

    source_lines = source_code.splitlines()
    all_nodes = []
    for m in methods:
        all_nodes.append((getattr(m.position, 'line', 999999) if m.position else 999999, m))
    for f in fields:
        all_nodes.append((getattr(f.position, 'line', 999999) if f.position else 999999, f))
    all_nodes.sort(key=lambda x: x[0])

    node_next_line = {}
    for i in range(len(all_nodes)):
        if i + 1 < len(all_nodes):
            node_next_line[id(all_nodes[i][1])] = all_nodes[i+1][0]
        else:
            node_next_line[id(all_nodes[i][1])] = None

    # Load Halstead metrics
    cyclomatic = float(row.get("cyclomatic", 0.0))
    lloc = float(row.get("lloc", 0.0))
    V = float(row.get("V", 0.0))
    D = float(row.get("D", 0.0))
    E = float(row.get("E", 0.0))
    B = float(row.get("B", 0.0))

    # Data usage
    n_field_accesses = 0
    n_method_calls = 0
    n_external_objects = 0

    method_names = {m.name for m in methods if hasattr(m, 'name')}
    field_names = {f.declarators[0].name for f in fields if hasattr(f, 'declarators') and f.declarators}

    for m in methods:
        if not m.body:
            continue
        for _, inv in m.filter(javalang.tree.MethodInvocation):
            n_method_calls += 1
            if inv.member not in method_names:
                n_external_objects += 1
        for _, ref in m.filter(javalang.tree.MemberReference):
            if ref.member in field_names:
                n_field_accesses += 1
            else:
                n_external_objects += 1

    total_accesses = max(1, n_field_accesses + n_method_calls)
    field_access_ratio = n_field_accesses / total_accesses
    method_call_ratio = n_method_calls / total_accesses
    data_usage_density = (n_field_accesses + n_method_calls) / max(1, len(methods))

    data_usage_feat = [
        min(1.0, n_field_accesses / 50.0),
        min(1.0, n_method_calls / 100.0),
        min(1.0, n_external_objects / 50.0),
        field_access_ratio,
        method_call_ratio,
        min(1.0, data_usage_density / 20.0)
    ]

    class_feat = [
        1.0, 0.0, 0.0,
        _norm(cyclomatic, "cyclomatic"), _norm(lloc, "lloc"), _norm(V, "V"),
        _norm(D, "D"), _norm(E, "E"), _norm(B, "B")
    ] + data_usage_feat

    # Embed class name
    class_name = cls.name if hasattr(cls, 'name') else str(row.get("code_name", "Unknown"))
    class_feat.extend(embedder.embed_identifier(class_name).tolist())

    node_features = [class_feat]
    src, dst, edge_types = [], [], []
    method_idx_map = {}
    field_idx_map = {}

    for i, m in enumerate(methods, start=1):
        if hasattr(m, 'name'):
            method_idx_map[m.name] = i

        stmts = sum(1 for _ in m.body) if m.body else 0
        cc = compute_cc(m)
        mloc = stmts * 1.5

        m_feat = [
            0.0, 1.0, 0.0,
            min(1.0, cc / 20.0), 0.0, 0.0, 0.0, 0.0,
            min(1.0, mloc / 200.0)
        ] + [0.0] * 6

        m_code = extract_node_code(m, source_lines, node_next_line.get(id(m)))
        if not m_code.strip():
            m_code = m.name if hasattr(m, 'name') else "method"
        m_feat.extend(embedder.embed_identifier(m_code).tolist())

        node_features.append(m_feat)
        src += [0, i]; dst += [i, 0]
        edge_types += [EDGE_TYPE["CONTAINS"], EDGE_TYPE["CONTAINS"]]

    for i, f in enumerate(fields, start=len(methods) + 1):
        f_name = f.declarators[0].name if hasattr(f, 'declarators') and f.declarators else "field"
        field_idx_map[f_name] = i

        f_feat = [0.0, 0.0, 1.0] + [0.0] * 6 + [0.0] * 6
        f_code = extract_node_code(f, source_lines, node_next_line.get(id(f)))
        if not f_code.strip():
            f_code = f_name
        f_feat.extend(embedder.embed_identifier(f_code).tolist())

        node_features.append(f_feat)
        src += [0, i]; dst += [i, 0]
        edge_types += [EDGE_TYPE["CONTAINS"], EDGE_TYPE["CONTAINS"]]

    # Rich Relational Edges
    for i, m in enumerate(methods, start=1):
        if not m.body: continue
        for _, inv in m.filter(javalang.tree.MethodInvocation):
            if inv.member in method_idx_map:
                target_idx = method_idx_map[inv.member]
                if target_idx != i:
                    src.append(i)
                    dst.append(target_idx)
                    edge_types.append(EDGE_TYPE["CALLS"])

        for _, ref in m.filter(javalang.tree.MemberReference):
            if ref.member in field_idx_map:
                target_idx = field_idx_map[ref.member]
                src.append(i)
                dst.append(target_idx)
                edge_types.append(EDGE_TYPE["ACCESSES_FIELD"])

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_type  = torch.tensor(edge_types, dtype=torch.long)  if edge_types else torch.zeros(0, dtype=torch.long)
    y = torch.tensor(int(row["smell_idx"]), dtype=torch.long)

    # Multi-label mask
    gc = int(row.get("GodClass", 0))
    fe = int(row.get("FeatureEnvy", 0))
    lm = int(row.get("LongMethod", 0))
    dc = int(row.get("DataClass", 0))
    co_smell_mask = torch.tensor([gc, fe, lm, dc], dtype=torch.float32)

    # Unnormalized Halstead
    halstead = torch.tensor([cyclomatic, lloc, V, D, E, B], dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "y": y,
        "co_smell_mask": co_smell_mask,
        "halstead": halstead
    }

# ──────────────────────────── Dataset ──────────────────────────────────────

class SmellDataset(Dataset):
    def __init__(self, df: pd.DataFrame, embedder: SemanticEmbedder):
        self.df = df.reset_index(drop=True)
        self.embedder = embedder
        self._graphs: List[Dict] = []
        self._build_all_graphs()

    def _build_all_graphs(self):
        logger.info(f"[Data] Building graphs for {len(self.df)} instances...")
        n = len(self.df)
        log_every = max(1, n // 10)
        
        self._graphs = []
        
        node_counts = []
        edge_counts = []
        method_counts = []
        field_counts = []
        
        for i, row in self.df.iterrows():
            graph = _build_ast_graph(row, str(row["source_code"]), self.embedder)
            graph["idx"] = i
            self._graphs.append(graph)
            
            x = graph["x"]
            edge_index = graph["edge_index"]
            node_counts.append(x.shape[0])
            edge_counts.append(edge_index.shape[1])
            method_counts.append(int(torch.sum(x[:, 1]).item()))
            field_counts.append(int(torch.sum(x[:, 2]).item()))
            
            if (i + 1) % log_every == 0:
                logger.info(f"[Data] Built {i+1}/{n} graphs...")
        
        logger.info(f"[Data] Graph construction complete. Total graphs: {len(self._graphs)}")
        logger.info(f"[Data] Graph characteristics summary:")
        logger.info(f"  Nodes:   min={np.min(node_counts)}, max={np.max(node_counts)}, mean={np.mean(node_counts):.2f}")
        logger.info(f"  Edges:   min={np.min(edge_counts)}, max={np.max(edge_counts)}, mean={np.mean(edge_counts):.2f}")
        logger.info(f"  Methods: min={np.min(method_counts)}, max={np.max(method_counts)}, mean={np.mean(method_counts):.2f}")
        logger.info(f"  Fields:  min={np.min(field_counts)}, max={np.max(field_counts)}, mean={np.mean(field_counts):.2f}")

    def __len__(self): return len(self._graphs)
    def __getitem__(self, idx): return self._graphs[idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"version": DATA_VERSION, "graphs": self._graphs}, path)
        logger.info(f"[Data] Saved {len(self._graphs)} graphs to {path}")

    @classmethod
    def load(cls, path: str) -> "SmellDataset":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls.__new__(cls)
        if isinstance(data, dict) and "graphs" in data:
            obj._graphs = data["graphs"]
        else:
            obj._graphs = data
        return obj

# ──────────────────────────── Preprocessing Pipeline ───────────────────────

def run_preprocessing(cfg: dict) -> Tuple["SmellDataset", "SmellDataset", "SmellDataset"]:
    from sklearn.model_selection import train_test_split

    proc_dir  = cfg["paths"]["processed_dir"]
    train_path = os.path.join(proc_dir, "train.pt")
    val_path   = os.path.join(proc_dir, "val.pt")
    test_path  = os.path.join(proc_dir, "test.pt")

    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        try:
            data = torch.load(train_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and data.get("version") == DATA_VERSION:
                probe = data["graphs"]
                if probe and "edge_type" in probe[0] and "halstead" in probe[0]:
                    logger.info(f"[Data] Processed .pt files found with matching version ({DATA_VERSION}) — loading cached graphs")
                    return SmellDataset.load(train_path), SmellDataset.load(val_path), SmellDataset.load(test_path)
        except Exception:
            pass
        
        logger.warning(
            f"[Data] Cached .pt files are missing or stale (expected version {DATA_VERSION}). "
            "Regenerating graph files."
        )

    dataset_type = cfg["dataset"].get("dataset_type", "smellycode")
    if dataset_type == "smellycode":
        csv_path = cfg["dataset"]["smellycode_csv"]
        df = load_smellycode(csv_path, nosmell_ratio=cfg["dataset"].get("nosmell_ratio", 3.0))
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Only 'smellycode' is supported.")

    embedder = SemanticEmbedder(cfg)

    seed = cfg["dataset"]["random_seed"]
    tr   = cfg["dataset"]["train_ratio"]
    vr   = cfg["dataset"]["val_ratio"]

    train_df, temp_df = train_test_split(df, test_size=1 - tr, stratify=df["smell_idx"], random_state=seed)
    val_size = vr / (1 - tr)
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_size, stratify=temp_df["smell_idx"], random_state=seed)

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