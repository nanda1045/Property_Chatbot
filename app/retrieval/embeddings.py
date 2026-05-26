from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from app.core.config import Settings

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)


class TextEmbedder(ABC):
    dimensions: int
    provider_name: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class LocalHashEmbedder(TextEmbedder):
    """Deterministic local embedder for offline prototype retrieval.

    This is intentionally simple: it creates a normalized hashing-vector from
    word unigrams and adjacent bigrams. It gives us repeatable Chroma ingestion
    without requiring API keys or model downloads. Production can replace this
    with OpenAI, Voyage, Cohere, or another embedding provider behind the same
    small interface.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.provider_name = "local_hash"

    def embed(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        features = Counter(tokens)
        features.update(
            f"{left}_{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        )

        vector = [0.0] * self.dimensions
        for feature, count in features.items():
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class SentenceTransformerEmbedder(TextEmbedder):
    """Local sentence-transformer embedder.

    The model is downloaded once into the configured cache path, then runs
    locally. The default `all-MiniLM-L6-v2` model yields 384-dimensional vectors.
    """

    def __init__(self, model_name: str, cache_folder: Path) -> None:
        from sentence_transformers import SentenceTransformer

        cache_folder.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.provider_name = "sentence_transformer"
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_folder),
            local_files_only=self._is_cached(cache_folder, model_name),
        )
        if hasattr(self.model, "get_embedding_dimension"):
            dimensions = self.model.get_embedding_dimension()
        else:
            dimensions = self.model.get_sentence_embedding_dimension()
        if dimensions is None:
            sample = self.model.encode(["dimension probe"], normalize_embeddings=True)[0]
            dimensions = len(sample)
        self.dimensions = int(dimensions)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [embedding.astype(float).tolist() for embedding in embeddings]

    @staticmethod
    def _is_cached(cache_folder: Path, model_name: str) -> bool:
        cache_key = f"models--{model_name.replace('/', '--')}"
        snapshot_root = cache_folder / cache_key / "snapshots"
        if not snapshot_root.exists():
            return False

        required_files = {
            "config.json",
            "modules.json",
            "model.safetensors",
            "tokenizer.json",
        }
        for snapshot_path in snapshot_root.iterdir():
            if snapshot_path.is_dir() and all(
                (snapshot_path / filename).exists() for filename in required_files
            ):
                return True
        return False


def build_embedder(settings: Settings) -> TextEmbedder:
    provider = settings.embedding_provider.lower()
    if provider == "sentence_transformer":
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            cache_folder=settings.embedding_cache_path,
        )
    if provider == "local_hash":
        return LocalHashEmbedder()
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
