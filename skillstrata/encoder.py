from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np


class TextEncoder:
    def __init__(
        self,
        model_id: str,
        model_name: str,
        cache_dir: str | Path,
        device: str,
        batch_size: int,
        max_sequence_length: int,
        query_prefix: str,
        cache_only: bool = False,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.query_prefix = query_prefix
        self.cache_only = cache_only
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self.cache_only:
            raise RuntimeError("embedding cache is missing while cache-only mode is enabled")
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[encoder] CUDA is unavailable; using CPU")
            device = "cpu"
        print(f"[encoder] loading {self.model_id} on {device}")
        self._model = SentenceTransformer(
            self.model_id,
            device=device,
            trust_remote_code=True,
        )
        self._model.max_seq_length = self.max_sequence_length
        return self._model

    def encode_candidates(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "candidates", "")

    def encode_queries(self, texts: list[str], split: str) -> np.ndarray:
        return self._encode(texts, split, self.query_prefix)

    def _encode(self, texts: list[str], key: str, prefix: str) -> np.ndarray:
        fingerprint_source = "\u0001".join(
            [self.model_id, str(self.max_sequence_length), prefix, *texts]
        )
        fingerprint = hashlib.sha256(
            fingerprint_source.encode("utf-8")
        ).hexdigest()[:16]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model_name)
        cache_path = self.cache_dir / f"{safe_name}_{key}_{len(texts)}_{fingerprint}.npy"
        if cache_path.exists():
            print(f"[encoder] cache hit: {cache_path.name}")
            return np.load(cache_path)
        if self.cache_only:
            raise FileNotFoundError(f"missing embedding cache: {cache_path}")

        model = self._load_model()
        embeddings = model.encode(
            [prefix + text for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)
        np.save(cache_path, embeddings)
        print(f"[encoder] saved: {cache_path.name} {embeddings.shape}")
        return embeddings
