"""
core/config.py — загрузка и кэширование конфигурации из YAML.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] | None = None
_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(config_path: str = _DEFAULT_PATH) -> Dict[str, Any]:
    """Загружает конфиг из файла и кэширует его в памяти процесса."""
    global _CONFIG
    with open(os.path.abspath(config_path), encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f) or {}
    return _CONFIG


def get_config() -> Dict[str, Any]:
    """Возвращает кэшированный конфиг, при необходимости загружает его."""
    if _CONFIG is None:
        load_config()
    return _CONFIG  # type: ignore[return-value]
