"""
models/semantic.py — семантическая метрика на базе FAISS + SentenceTransformers.

Регистрируется в глобальном реестре под именем "faiss_centroid".
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import SemanticMetric, semantic_registry

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "corpus", "ethical_corpus.txt"
)
_LRU_CACHE_SIZE = 128


def _load_corpus(path: str) -> list[str]:
    abspath = os.path.abspath(path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(
            f"Эталонный корпус не найден: {abspath}\n"
            "Создайте файл corpus/ethical_corpus.txt с фразами (по одной на строку)."
        )
    with open(abspath, encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not lines:
        raise ValueError("Корпус пуст. Добавьте фразы в corpus/ethical_corpus.txt.")
    return lines


@semantic_registry.register("faiss_centroid")
class FaissCentroidSemantic(SemanticMetric):
    """
    Ssem = cosine_similarity(embed(text), centroid(corpus)).

    Алгоритм:
      1. При инициализации строит FAISS-индекс поверх эмбеддингов корпуса.
      2. Вычисляет и нормализует центроид корпуса.
      3. При каждом predict(): нормализует эмбеддинг текста, считает dot-product
         с центроидом (= cosine similarity для нормализованных векторов),
         дополнительно возвращает top-k ближайших фраз из корпуса.
    """

    def __init__(
        self,
        corpus_path: str = _DEFAULT_CORPUS_PATH,
        device: Optional[str] = None,
    ) -> None:
        import torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[FaissCentroidSemantic] Загрузка модели '{_MODEL_NAME}'...")
        self._model = SentenceTransformer(_MODEL_NAME, device=self._device)
        self._dim = self._model.get_sentence_embedding_dimension()

        print(f"[FaissCentroidSemantic] Загрузка корпуса из '{corpus_path}'...")
        self._corpus = _load_corpus(corpus_path)
        corpus_embeddings = self._embed_batch(self._corpus)

        self._centroid = self._compute_centroid(corpus_embeddings)
        self._index = self._build_faiss_index(corpus_embeddings)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        print(
            f"[FaissCentroidSemantic] Готово. "
            f"Корпус: {len(self._corpus)} фраз, dim={self._dim}"
        )

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "faiss_centroid"

    def predict(self, text: str, top_k: int = 3) -> dict:
        """
        Возвращает:
            {
                "ssem": float,          # ∈ [0, 1]
                "top_k": [
                    {"phrase": str, "similarity": float}, ...
                ]
            }
        """
        if not text or not text.strip():
            return {"ssem": 0.0, "top_k": []}

        vec = self._get_embedding(text).reshape(1, -1)
        ssem = float(np.clip(np.dot(vec[0], self._centroid), 0.0, 1.0))
        nearest = self._search_nearest(vec, top_k)

        return {"ssem": ssem, "top_k": nearest}

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Возвращает нормализованный эмбеддинг с LRU-кэшем."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        emb = self._embed_batch([text])[0]
        self._cache[text] = emb
        if len(self._cache) > _LRU_CACHE_SIZE:
            self._cache.popitem(last=False)
        return emb

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return (centroid / norm).astype("float32") if norm > 1e-8 else centroid.astype("float32")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(self._dim)
        index.add(embeddings)
        return index

    def _search_nearest(self, vec: np.ndarray, top_k: int) -> list[dict]:
        k = min(top_k, len(self._corpus))
        similarities, indices = self._index.search(vec, k)
        return [
            {"phrase": self._corpus[int(idx)], "similarity": float(sim)}
            for sim, idx in zip(similarities[0], indices[0])
        ]
