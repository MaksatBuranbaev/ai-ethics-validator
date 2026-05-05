"""
models/toxicity.py — классификатор токсичности на базе BERT.

Регистрируется в глобальном реестре под именем "bert_toxicity".
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import ToxicityMetric, toxicity_registry

_MODEL_NAME = "s-nlp/russian_toxicity_classifier"


@toxicity_registry.register("bert_toxicity")
class BertToxicityClassifier(ToxicityMetric):
    """
    Вычисляет Ptox ∈ [0, 1] — вероятность токсичного класса.

    Дополнительно реализует XAI-метод explain() методом leave-one-out:
    importance(token) = Ptox(original) − Ptox(text без токена).
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BertToxicityClassifier] Загрузка '{_MODEL_NAME}' на {self._device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()
        self._toxic_idx = self._find_toxic_class_index()
        print(f"[BertToxicityClassifier] Готово. Индекс toxic-класса: {self._toxic_idx}")

    # ------------------------------------------------------------------
    # BaseMetric interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "bert_toxicity"

    @torch.no_grad()
    def predict(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        inputs = self._encode(text)
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self._toxic_idx].item()

    # ------------------------------------------------------------------
    # XAI
    # ------------------------------------------------------------------

    def explain(self, text: str) -> list[dict]:
        """
        Leave-one-out XAI: для каждого токена вычисляет, насколько
        его отсутствие снижает (или повышает) токсичность.

        Возвращает список {"token": str, "importance": float}.
        Положительное значение → токен увеличивает токсичность.
        """
        if not text or not text.strip():
            return []

        tokens = self._tokenizer.tokenize(text)
        if not tokens:
            return []

        base_score = self.predict(text)

        results = []
        for i, token in enumerate(tokens):
            masked_tokens = tokens[:i] + tokens[i + 1:]
            masked_text = (
                self._tokenizer.convert_tokens_to_string(masked_tokens)
                if masked_tokens
                else ""
            )
            masked_score = self.predict(masked_text)
            results.append({
                "token": token,
                "importance": round(base_score - masked_score, 6),
            })

        return results

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

    def _find_toxic_class_index(self) -> int:
        id2label = self._model.config.id2label
        for idx, label in id2label.items():
            if "toxic" in label.lower():
                return int(idx)
        raise ValueError(
            f"Класс 'toxic' не найден в {_MODEL_NAME}. "
            f"Доступные классы: {list(id2label.values())}"
        )
