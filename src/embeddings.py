import os
import json
import torch
import hashlib
import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger("SmellRL.embeddings")

# ─── ABLATION TOGGLE OVERRIDE ───
# Set to False to completely skip loading the heavy transformers model
USE_EMBEDDINGS_MODEL = True
# ────────────────────────────────

class SyntheticEmbedder:
    """Generates deterministic 768-dim pseudo-embeddings via hashing."""
    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode('utf-8')).digest()
        # Seed numpy with hash to create a deterministic vector
        seed = int.from_bytes(h[:4], byteorder='little')
        rng = np.random.RandomState(seed)
        return rng.normal(0, 0.1, self.dim).astype(np.float32)

class SemanticEmbedder:
    """Wraps GraphCodeBERT to generate semantic embeddings with disk caching."""
    def __init__(self, cfg: dict):
        self.dim = cfg["semantic"]["embedding_dim"]
        self.cache_dir = cfg["semantic"]["cache_dir"]
        self.use_cache = cfg["semantic"]["use_cache"]
        self.max_length = cfg["semantic"]["max_length"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.synthetic = SyntheticEmbedder(self.dim)
        
        # ─── ABLATION TOGGLE OVERRIDE CHECK ───
        if not USE_EMBEDDINGS_MODEL:
            logger.info("[Embeddings] Ablation active: Skipping GraphCodeBERT loading completely.")
            self.has_transformers = False
            return
        # ──────────────────────────────────────
        
        # If USE_EMBEDDINGS_MODEL=True, transformers MUST be available.
        # We intentionally do NOT catch ImportError here — a missing transformers
        # install should fail loudly rather than silently fall back to synthetic
        # embeddings and produce misleading results.
        from transformers import AutoTokenizer, AutoModel
        model_name = cfg["semantic"]["model_name"]
        logger.info(f"[Embeddings] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.has_transformers = True
            
    def _get_cache_path(self, text: str) -> str:
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    @torch.no_grad()
    def embed_identifier(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        cache_path = self._get_cache_path(text)
        if self.use_cache and os.path.exists(cache_path):
            return np.load(cache_path)

        if not self.has_transformers:
            # Reached only when USE_EMBEDDINGS_MODEL=False (explicit ablation).
            emb = self.synthetic.embed(text)
        else:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Pool using the CLS token
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

        if self.use_cache:
            np.save(cache_path, emb)
        return emb