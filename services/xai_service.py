"""
services/xai_service.py — сервис объяснимости (XAI).

Выделен в отдельный класс, чтобы в будущем можно было добавить
другие методы XAI (SHAP, LIME, attention-based) без изменения
EvaluationService или маршрутов API.
"""

from __future__ import annotations

from typing import List, Dict, Any

from models.base import ModelContainer, get_model_container
from core.preprocessor import get_preprocessor


class XAIService:
    """
    Предоставляет методы объяснимости для метрик.

    Сейчас реализует только leave-one-out для токсичности.
    Новые методы (SHAP, attention) добавляются сюда же.
    """

    def __init__(self, container: ModelContainer | None = None) -> None:
        self._container = container or get_model_container()
        self._preprocessor = get_preprocessor()

    def explain_toxicity(self, text: str) -> List[Dict[str, Any]]:
        """
        Возвращает важность токенов для предсказания токсичности.

        Делегирует методу ToxicityMetric.explain() — если конкретная
        модель не реализует XAI, graceful degradation возвращает [].

        Формат ответа: [{"token": str, "importance": float}, ...]
        Положительное importance → токен повышает токсичность.
        """
        cleaned = self._preprocessor.preprocess(text)
        return self._container.toxicity.explain(cleaned)
