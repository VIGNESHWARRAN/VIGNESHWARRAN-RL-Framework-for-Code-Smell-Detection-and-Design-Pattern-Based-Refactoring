import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

from src.embeddings import SemanticEmbedder

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
    return float(np.clip(val, 0, _CK_MAX.get(key, 100.0)) / _CK_MAX.get(key, 100.0))

# ──────────────────────────── AST Helpers ──────────────────────────────────

def compute_cc(method_node) -> int:
    import javalang
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
    # Ensure we don't extract too many lines (e.g., if next node is far)
    # 50 lines max to prevent huge strings if parsing is weird
    end = min(end, start + 50)
    return "\n".join(source_lines[start:end])

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

def check_single_code(code_str: str) -> bool:
    if not code_str or code_str.strip() == "":
        return False
    try:
        import javalang
        tree = javalang.parse.parse(code_str)
        return bool(tree.types)
    except Exception:
        return False

def load_mlcq(csv_path: str, nosmell_ratio: float = 1.0) -> pd.DataFrame:
    logger.info(f"[Data] Loading MLCQ CSV from {csv_path}")
    
    # Cache path for 100% parseable AST rows
    cache_csv_path = csv_path.replace(".csv", "_parseable.csv")
    
    if os.path.exists(cache_csv_path):
        logger.info(f"[Data] Found cached parseable CSV: {cache_csv_path}")
        df = pd.read_csv(cache_csv_path)
    else:
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
        
        # ─── AST Parse Verification (Task 1 & Task 2) with Multiprocessing ───
        total_loaded = len(df)
        has_code_col = "source_code" in df.columns
        
        if has_code_col:
            codes = [str(row.get("source_code", "")) if pd.notna(row.get("source_code")) else "" for _, row in df.iterrows()]
        else:
            codes = []
            
        rows_with_code = sum(1 for c in codes if c.strip() != "")
        
        logger.info(f"[Data] Pre-validating {len(df)} rows for AST parseability using multiprocessing...")
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(check_single_code, codes, chunksize=100))
            
        parseable_indices = [i for i, ok in enumerate(results) if ok]
        parse_success = len(parseable_indices)
        parse_failed = rows_with_code - parse_success
        no_code = total_loaded - rows_with_code
        
        logger.info(f"[Data] Pre-validation parsing stats:")
        logger.info(f"  Total CSV rows loaded: {total_loaded}")
        logger.info(f"  Rows with source code: {rows_with_code}")
        logger.info(f"  Successfully parsed AST: {parse_success}")
        logger.info(f"  Failed AST parsing (Syntax errors): {parse_failed}")
        logger.info(f"  Missing source code: {no_code}")
        
        df = df.iloc[parseable_indices].reset_index(drop=True)
        
        # Save cache
        df.to_csv(cache_csv_path, index=False)
        logger.info(f"[Data] Saved cached parseable CSV to: {cache_csv_path}")

    # ─── Class Balancing on Parsed Trees (Task 3) ───
    df_smells = df[df["smell_label"] != "NoSmell"]
    df_nosmells = df[df["smell_label"] == "NoSmell"]

    logger.info("[Data] Class distribution of parsed ASTs before downsampling:")
    for smell_cls in SMELL_CLASSES:
        count = len(df[df["smell_label"] == smell_cls])
        logger.info(f"  {smell_cls}: {count}")

    if not df_nosmells.empty and not df_smells.empty:
        max_smell_count = df_smells["smell_label"].value_counts().max()
        target_nosmell_count = int(max_smell_count * nosmell_ratio)
        if len(df_nosmells) > target_nosmell_count:
            df_nosmells_downsampled = df_nosmells.sample(n=target_nosmell_count, random_state=42)
            df = pd.concat([df_smells, df_nosmells_downsampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info("[Data] Class distribution after downsampling:")
    for smell_cls in SMELL_CLASSES:
        count = len(df[df["smell_label"] == smell_cls])
        logger.info(f"  {smell_cls}: {count}")

    ck_cols = ["wmc", "dit", "noc", "cbo", "rfc", "loc", "n_methods", "n_fields"]
    for col in ck_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in ck_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df

# ──────────────────────────── Graph Builder ────────────────────────────────

def _build_ast_graph(row: pd.Series, source_code: str, embedder: SemanticEmbedder) -> Optional[Dict]:
    try:
        import javalang
        tree = javalang.parse.parse(source_code)
        if not tree.types:
            return None

        cls = tree.types[0]
        methods = list(cls.methods) if hasattr(cls, "methods") else []
        fields  = list(cls.fields)  if hasattr(cls, "fields")  else []
        
        # Prepare for code body extraction
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

        wmc = float(row.get("wmc", len(methods) * 3))
        dit = float(row.get("dit", 0))
        noc = float(row.get("noc", 0))
        cbo = float(row.get("cbo", 0))
        rfc = float(row.get("rfc", 0))
        loc = float(row.get("loc", len(source_lines)))

        # ─── Data Usage Features ───
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
            _norm(wmc, "wmc"), _norm(dit, "dit"), _norm(noc, "noc"),
            _norm(cbo, "cbo"), _norm(rfc, "rfc"), _norm(loc, "loc"),
        ] + data_usage_feat
        
        class_name = cls.name if hasattr(cls, 'name') else str(row.get("code_name", "Unknown"))
        class_feat.extend(embedder.embed_identifier(class_name).tolist())
        
        node_features = [class_feat]
        src, dst = [], []
        
        # Maps for rich edges
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
                min(1.0, mloc / 200.0),
            ] + [0.0] * 6 # Pad data_usage
            
            # Embed actual code body instead of just name
            m_code = extract_node_code(m, source_lines, node_next_line.get(id(m)))
            if not m_code.strip():
                m_code = m.name if hasattr(m, 'name') else "method"
            m_feat.extend(embedder.embed_identifier(m_code).tolist())
            
            node_features.append(m_feat)
            src += [0, i]; dst += [i, 0]

        for i, f in enumerate(fields, start=len(methods) + 1):
            f_name = f.declarators[0].name if hasattr(f, 'declarators') and f.declarators else "field"
            field_idx_map[f_name] = i
            
            f_feat = [0.0, 0.0, 1.0] + [0.0] * 6 + [0.0] * 6 # pad CK and data_usage
            
            # Embed field declaration code
            f_code = extract_node_code(f, source_lines, node_next_line.get(id(f)))
            if not f_code.strip():
                f_code = f_name
            f_feat.extend(embedder.embed_identifier(f_code).tolist())
            
            node_features.append(f_feat)
            src += [0, i]; dst += [i, 0]

        # ─── Add Richer Edges (CALLS, ACCESSES_FIELD) ───
        for i, m in enumerate(methods, start=1):
            if not m.body: continue
            
            # CALLS
            for _, inv in m.filter(javalang.tree.MethodInvocation):
                if inv.member in method_idx_map:
                    target_idx = method_idx_map[inv.member]
                    if target_idx != i: # avoid self loop
                        src.append(i)
                        dst.append(target_idx)
            
            # ACCESSES_FIELD
            for _, ref in m.filter(javalang.tree.MemberReference):
                if ref.member in field_idx_map:
                    target_idx = field_idx_map[ref.member]
                    src.append(i)
                    dst.append(target_idx)

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        y = torch.tensor(int(row["smell_idx"]), dtype=torch.long)
        return {"x": x, "edge_index": edge_index, "y": y}

    except Exception as e:
        logger.error(f"[Data] Unexpected error building AST graph for class {row.get('code_name', 'Unknown')}: {e}")
        raise e

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
            if graph is None:
                raise ValueError(f"Failed to build AST graph for row {i} even though it was validated as parseable.")
            
            graph["idx"] = i
            self._graphs.append(graph)
            
            # Record graph stats for visual logging (Task 4)
            x = graph["x"]
            edge_index = graph["edge_index"]
            node_counts.append(x.shape[0])
            edge_counts.append(edge_index.shape[1])
            
            # Node features dimensions 0, 1, 2 represent class, method, field nodes respectively
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