"""
models/empathy.py — прокси-метрика эмпатии на базе сентимент-анализа.

Регистрируется в глобальном реестре под именем "sentiment_proxy".
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import EmpathyMetric, empathy_registry

_MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"


@empathy_registry.register("sentiment_proxy")
class SentimentProxyEmpathy(EmpathyMetric):
    """
    Прокси-метрика эмпатии: Pemp = P(positive | text).

    Эмпатичный ответ статистически коррелирует с позитивным тоном,
    поэтому вероятность positive-класса используется как быстрый прокси.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SentimentProxyEmpathy] Загрузка '{_MODEL_NAME}' на {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()
        self._positive_idx = self._find_positive_class_index()
        print(f"[SentimentProxyEmpathy] Готово. Индекс positive-класса: {self._positive_idx}")

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "sentiment_proxy"

    @torch.no_grad()
    def predict(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        inputs = self._encode(text)
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self._positive_idx].item()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> dict:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def _find_positive_class_index(self) -> int:
        id2label = self._model.config.id2label
        for idx, label in id2label.items():
            if "pos" in label.lower():
                return int(idx)
        raise ValueError(
            f"Класс 'positive' не найден в {_MODEL_NAME}. "
            f"Доступные классы: {list(id2label.values())}"
        )
