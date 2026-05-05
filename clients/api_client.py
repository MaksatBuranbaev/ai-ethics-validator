"""
clients/api_client.py — HTTP-клиент для Streamlit UI.

ApiClient скрывает детали транспорта (requests, URL, таймауты)
и предоставляет типизированный интерфейс для вызова серверных методов.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from core.config import get_config

logger = logging.getLogger(__name__)


class ApiClient:
    """
    Клиент для взаимодействия с FastAPI-сервером.

    Все методы маппируются 1-к-1 на REST-эндпоинты:
        evaluate()  → POST /evaluate
        pareto()    → POST /pareto
        explain()   → POST /explain
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30) -> None:
        cfg = get_config()
        self._base_url = (base_url or cfg.get("api", {}).get("url", "http://localhost:8000")).rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def evaluate(self, text: str, tau: float = 0.3) -> Dict[str, Any]:
        return self._post("evaluate", {"text": text, "tau": tau})

    def pareto(self, candidates: List[str], tau: float = 0.3) -> Dict[str, Any]:
        return self._post("pareto", {"candidates": candidates, "tau": tau})

    def explain(self, text: str) -> Dict[str, Any]:
        return self._post("explain", {"text": text})

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}/{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(f"Не удалось подключиться к серверу: {url}") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Сервер вернул ошибку {exc.response.status_code}: {exc.response.text}"
            ) from exc


# ---------------------------------------------------------------------------
# Синглтон
# ---------------------------------------------------------------------------

_client: Optional[ApiClient] = None


def get_api_client() -> ApiClient:
    """Возвращает глобальный экземпляр ApiClient."""
    global _client
    if _client is None:
        _client = ApiClient()
    return _client
