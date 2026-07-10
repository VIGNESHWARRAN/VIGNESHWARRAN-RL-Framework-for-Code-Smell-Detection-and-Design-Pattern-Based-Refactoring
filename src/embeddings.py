import os
import torch
import hashlib
import logging
import numpy as np

# We enforce that transformers MUST be installed. Any missing dependency crashes the process immediately.
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("SmellRL.embeddings")

class SemanticEmbedder:
    """Wraps GraphCodeBERT to generate semantic embeddings with disk caching."""
    def __init__(self, cfg: dict):
        self.dim = cfg["semantic"]["embedding_dim"]
        self.cache_dir = cfg["semantic"]["cache_dir"]
        self.use_cache = cfg["semantic"]["use_cache"]
        self.max_length = cfg["semantic"]["max_length"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        model_name = cfg["semantic"]["model_name"]
        logger.info(f"[Embeddings] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
            
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

        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Pool using the CLS token
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

        if self.use_cache:
            np.save(cache_path, emb)
        return emb