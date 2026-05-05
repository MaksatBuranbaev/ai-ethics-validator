"""
services/evaluation_service.py — бизнес-логика оценки текстов.

EvaluationService является основным use-case слоем:
  - принимает сырой текст,
  - прогоняет его через предобработку и модели,
  - делегирует агрегацию aggregator.py,
  - опционально сохраняет результат в БД.

Не содержит деталей HTTP, моделей или SQL — только оркестрацию.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aggregator import DEFAULT_TAU, EvalResult, aggregate, pareto_front
from core.config import get_config
from core.preprocessor import get_preprocessor
from models.base import ModelContainer, get_model_container

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Оркестрирует пайплайн оценки одного или нескольких текстов.

    Зависимости передаются через конструктор, что облегчает тестирование
    (можно подставить mock-контейнер с фиктивными моделями).
    """

    def __init__(
        self,
        container: Optional[ModelContainer] = None,
        default_tau: Optional[float] = None
    ) -> None:
        cfg = get_config()
        eval_cfg = cfg.get("evaluation", {})

        self._container = container or get_model_container()
        self._preprocessor = get_preprocessor()
        self._default_tau: float = default_tau if default_tau is not None else eval_cfg.get(
            "default_tau", DEFAULT_TAU
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        text: str,
        tau_tox: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает один текст.

        Возвращает словарь, совместимый с EvaluateResponse (Pydantic-схема в routes.py).
        """
        tau = tau_tox if tau_tox is not None else self._default_tau

        cleaned = self._preprocessor.preprocess(text)

        ptox = self._container.compute_ptox(cleaned)
        pemp = self._container.compute_pemp(cleaned)
        sem = self._container.compute_ssem_full(cleaned, top_k=3)

        result = aggregate(
            ptox=ptox,
            pemp=pemp,
            ssem=sem["ssem"],
            tau_tox=tau,
            top_k_phrases=sem["top_k"],
        )

        return result.to_dict()

    def evaluate_multiple(
        self,
        texts: List[str],
        tau_tox: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Оценивает список текстов и вычисляет Парето-фронт.

        Возвращает {"results": [...], "pareto_indices": [...]}.
        """
        results = [self.evaluate(t, tau_tox=tau_tox) for t in texts]
        candidates = [{"ptox": r["ptox"], "pemp": r["pemp"], "ssem": r["ssem"]} for r in results]
        pareto_idx = pareto_front(candidates)
        return {"results": results, "pareto_indices": pareto_idx}

    def explain_toxicity(self, text: str) -> List[Dict[str, Any]]:
        """
        XAI: возвращает список {"token": str, "importance": float}.

        Делегирует методу ToxicityMetric.explain() — если конкретная
        модель не реализует XAI, возвращается пустой список.
        """
        cleaned = self._preprocessor.preprocess(text)
        return self._container.toxicity.explain(cleaned)
