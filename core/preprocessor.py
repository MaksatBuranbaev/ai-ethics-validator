"""
core/preprocessor.py — нормализация текста перед подачей в модели.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


class TextPreprocessor:
    """
    Применяет последовательность нормализаций к тексту.

    Все опции управляются через конфиг (секция ``preprocessing`` в config.yaml).
    """

    def __init__(self, config: dict) -> None:
        self._enabled: bool            = config.get("enabled", True)
        self._lower_case: bool         = config.get("lower_case", False)
        self._remove_extra_spaces: bool = config.get("remove_extra_spaces", True)
        self._fix_unicode: bool        = config.get("fix_unicode", True)

    def preprocess(self, text: str) -> str:
        if not self._enabled or not text:
            return text

        if self._fix_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self._lower_case:
            text = text.lower()

        if self._remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()

        return text


# ---------------------------------------------------------------------------
# Синглтон
# ---------------------------------------------------------------------------

_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor() -> TextPreprocessor:
    """Возвращает глобальный экземпляр TextPreprocessor."""
    global _preprocessor
    if _preprocessor is None:
        from core.config import get_config
        cfg = get_config().get("preprocessing", {})
        _preprocessor = TextPreprocessor(cfg)
    return _preprocessor
