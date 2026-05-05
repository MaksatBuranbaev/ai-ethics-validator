"""
api/dependencies.py — провайдеры зависимостей для FastAPI (Dependency Injection).

Каждая функция-провайдер возвращает синглтон-сервис, созданный при первом вызове.
Синглтоны хранятся на уровне модуля, а не в FastAPI-стейте, что позволяет
переиспользовать их между запросами без overhead.
"""

from __future__ import annotations

from functools import lru_cache

from services.evaluation_service import EvaluationService


@lru_cache(maxsize=1)
def get_evaluation_service() -> EvaluationService:
    """Возвращает единственный экземпляр EvaluationService."""
    return EvaluationService()
