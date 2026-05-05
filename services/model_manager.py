"""
services/model_manager.py — инициализация моделей из конфигурации.

Единственная точка, где config.yaml встречается с реестром моделей.
"""

from __future__ import annotations

import logging
from typing import Optional

import models  # noqa: F401 — регистрирует все модели через __init__.py
from models.base import get_model_container
from core.config import get_config

logger = logging.getLogger(__name__)

_models_initialized = False


def init_models_from_config(device: Optional[str] = None) -> None:
    """
    Читает секцию ``models`` из config.yaml и инициализирует
    активные модели в глобальном ModelContainer.

    Идемпотентна: повторный вызов ничего не делает.
    """
    global _models_initialized
    if _models_initialized:
        return

    cfg = get_config()
    models_cfg = cfg.get("models", {})
    device = device or cfg.get("device")  # опциональный ключ device в config.yaml
    init_kwargs = {"device": device} if device else {}

    container = get_model_container()

    if "toxicity" in models_cfg:
        logger.info("Инициализация модели токсичности: %s", models_cfg["toxicity"])
        container.set_toxicity(models_cfg["toxicity"], **init_kwargs)

    if "empathy" in models_cfg:
        logger.info("Инициализация модели эмпатии: %s", models_cfg["empathy"])
        container.set_empathy(models_cfg["empathy"], **init_kwargs)

    if "semantic" in models_cfg:
        logger.info("Инициализация семантической модели: %s", models_cfg["semantic"])
        semantic_kwargs = {**init_kwargs}
        corpus_path = cfg.get("corpus", {}).get("path")
        if corpus_path:
            semantic_kwargs["corpus_path"] = corpus_path
        container.set_semantic(models_cfg["semantic"], **semantic_kwargs)

    _models_initialized = True
    logger.info("Все модели успешно инициализированы.")
