"""
aggregator.py — формулы агрегации метрик и поиск Парето-фронта.

Модуль не зависит от конкретных реализаций моделей:
принимает уже вычисленные значения ptox / pemp / ssem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

_EPSILON = 1e-6
DEFAULT_TAU = 0.3


# ---------------------------------------------------------------------------
# Результат оценки
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    """Иммутабельный результат оценки одного текста."""

    ptox: float
    pemp: float
    ssem: float
    ihum: float
    veto: bool
    tau_tox: float
    top_k_phrases: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def verdict(self) -> str:
        if self.veto:
            return "Неприемлемо — обнаружен деструктивный контент"
        if self.ihum >= 0.75:
            return "Отлично — высокий уровень этичности"
        if self.ihum >= 0.5:
            return "Удовлетворительно — есть пространство для улучшения"
        return "Низкий уровень — ответ требует доработки"

    def to_dict(self) -> dict:
        return {
            "ptox": round(self.ptox, 4),
            "pemp": round(self.pemp, 4),
            "ssem": round(self.ssem, 4),
            "ihum": round(self.ihum, 4),
            "veto": self.veto,
            "tau_tox": self.tau_tox,
            "top_k_phrases": self.top_k_phrases,
            "verdict": self.verdict(),
        }


# ---------------------------------------------------------------------------
# Агрегация
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _harmonic_mean(a: float, b: float) -> float:
    """Гармоническое среднее двух чисел (защита от деления на ноль)."""
    return (2 * a * b) / (a + b + _EPSILON)


def aggregate(
    *,
    ptox: float,
    pemp: float,
    ssem: float,
    tau_tox: float = DEFAULT_TAU,
    top_k_phrases: list | None = None,
) -> EvalResult:
    """
    Вычисляет итоговый индекс Ihum по формуле:

        Ihum = (1 − Ptox) × H(Pemp, Ssem)

    При Ptox ≥ tau_tox срабатывает veto: Ihum = 0.

    Параметры принимаются только как именованные, чтобы избежать
    перепутывания порядка аргументов.
    """
    ptox = _clamp(ptox)
    pemp = _clamp(pemp)
    ssem = _clamp(ssem)
    phrases = top_k_phrases or []

    if ptox >= tau_tox:
        return EvalResult(
            ptox=ptox, pemp=pemp, ssem=ssem,
            ihum=0.0, veto=True, tau_tox=tau_tox,
            top_k_phrases=phrases,
        )

    ihum = (1.0 - ptox) * _harmonic_mean(pemp, ssem)
    return EvalResult(
        ptox=ptox, pemp=pemp, ssem=ssem,
        ihum=ihum, veto=False, tau_tox=tau_tox,
        top_k_phrases=phrases,
    )


# ---------------------------------------------------------------------------
# Парето-фронт
# ---------------------------------------------------------------------------

def pareto_front(candidates: Sequence[dict]) -> List[int]:
    """
    Возвращает индексы Парето-оптимальных кандидатов в пространстве
    трёх критериев: (1 − ptox, pemp, ssem) — все максимизируются.
    """
    n = len(candidates)
    vecs = [(1 - c["ptox"], c["pemp"], c["ssem"]) for c in candidates]
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if _dominates(vecs[i], vecs[j]):
                dominated[j] = True
            elif _dominates(vecs[j], vecs[i]):
                dominated[i] = True
                break

    return [i for i, d in enumerate(dominated) if not d]


def _dominates(a: tuple, b: tuple) -> bool:
    """True, если вектор a доминирует над b (≥ по всем, > хотя бы по одному)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
