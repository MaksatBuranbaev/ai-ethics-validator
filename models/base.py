"""
models/base.py — абстрактные интерфейсы метрик и реестр моделей.

Архитектура:
  * BaseMetric — общий интерфейс для всех метрик.
  * ToxicityMetric / EmpathyMetric / SemanticMetric — типизированные подклассы.
  * ModelRegistry — обобщённый реестр + фабрика экземпляров.
  * ModelContainer — синглтон-контейнер активных моделей (DI-совместимый).

Добавление новой модели:
  1. Создайте класс, наследующий нужный абстрактный тип.
  2. Украсьте декоратором @registry.register("my_name").
  3. Укажите "my_name" в config.yaml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

# ---------------------------------------------------------------------------
# Абстрактные интерфейсы
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """Общий интерфейс для всех метрик."""

    @abstractmethod
    def predict(self, text: str, **kwargs) -> Any:
        """Основной метод предсказания."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Человекочитаемый идентификатор метрики."""


class ToxicityMetric(BaseMetric, ABC):
    """Метрика токсичности: возвращает Ptox ∈ [0, 1]."""

    @abstractmethod
    def predict(self, text: str) -> float: ...

    def explain(self, text: str) -> list[dict]:
        """
        Необязательный метод XAI.
        Возвращает список {"token": str, "importance": float}.
        По умолчанию — пустой список (graceful degradation).
        """
        return []


class EmpathyMetric(BaseMetric, ABC):
    """Метрика эмпатии: возвращает Pemp ∈ [0, 1]."""

    @abstractmethod
    def predict(self, text: str) -> float: ...


class SemanticMetric(BaseMetric, ABC):
    """
    Семантическая метрика.
    Возвращает {"ssem": float, "top_k": list[{"phrase": str, "similarity": float}]}.
    """

    @abstractmethod
    def predict(self, text: str, top_k: int = 3) -> dict: ...


# ---------------------------------------------------------------------------
# Обобщённый реестр
# ---------------------------------------------------------------------------

M = TypeVar("M", bound=BaseMetric)


class ModelRegistry(Generic[M]):
    """
    Реестр + фабрика для одного типа метрик.

    Пример:
        toxicity_registry = ModelRegistry[ToxicityMetric]("toxicity")

        @toxicity_registry.register("bert_toxicity")
        class BertToxicity(ToxicityMetric): ...

        model = toxicity_registry.build("bert_toxicity", device="cpu")
    """

    def __init__(self, metric_type: str) -> None:
        self._metric_type = metric_type
        self._registry: Dict[str, Type[M]] = {}

    # ------------------------------------------------------------------
    # Регистрация
    # ------------------------------------------------------------------

    def register(self, name: str):
        """Декоратор для регистрации класса модели под именем ``name``."""
        def decorator(cls: Type[M]) -> Type[M]:
            if name in self._registry:
                raise ValueError(
                    f"[{self._metric_type}] Имя '{name}' уже зарегистрировано "
                    f"за классом {self._registry[name].__name__}."
                )
            self._registry[name] = cls
            return cls
        return decorator

    # ------------------------------------------------------------------
    # Фабрика
    # ------------------------------------------------------------------

    def build(self, name: str, **init_kwargs) -> M:
        """Создаёт и возвращает новый экземпляр зарегистрированной модели."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"[{self._metric_type}] Неизвестное имя модели: '{name}'. "
                f"Доступны: {available}"
            )
        return self._registry[name](**init_kwargs)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @property
    def available(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# ---------------------------------------------------------------------------
# Глобальные реестры (по одному на тип метрики)
# ---------------------------------------------------------------------------

toxicity_registry: ModelRegistry[ToxicityMetric] = ModelRegistry("toxicity")
empathy_registry:  ModelRegistry[EmpathyMetric]  = ModelRegistry("empathy")
semantic_registry: ModelRegistry[SemanticMetric] = ModelRegistry("semantic")


# ---------------------------------------------------------------------------
# Контейнер активных моделей
# ---------------------------------------------------------------------------

class ModelContainer:
    """
    Хранит единственный активный экземпляр каждой метрики.

    Используется как синглтон через ``get_model_container()``,
    но может быть заменён тестовым дублёром без изменения кода.
    """

    def __init__(self) -> None:
        self._toxicity: Optional[ToxicityMetric] = None
        self._empathy:  Optional[EmpathyMetric]  = None
        self._semantic: Optional[SemanticMetric] = None

    # ------------------------------------------------------------------
    # Сеттеры (устанавливают модель по имени из реестра)
    # ------------------------------------------------------------------

    def set_toxicity(self, name: str, **kwargs) -> None:
        self._toxicity = toxicity_registry.build(name, **kwargs)

    def set_empathy(self, name: str, **kwargs) -> None:
        self._empathy = empathy_registry.build(name, **kwargs)

    def set_semantic(self, name: str, **kwargs) -> None:
        self._semantic = semantic_registry.build(name, **kwargs)

    # ------------------------------------------------------------------
    # Геттеры (поднимают RuntimeError, если модель не инициализирована)
    # ------------------------------------------------------------------

    @property
    def toxicity(self) -> ToxicityMetric:
        if self._toxicity is None:
            raise RuntimeError(
                "Модель токсичности не инициализирована. "
                "Вызовите ModelContainer.set_toxicity() или init_models_from_config()."
            )
        return self._toxicity

    @property
    def empathy(self) -> EmpathyMetric:
        if self._empathy is None:
            raise RuntimeError(
                "Модель эмпатии не инициализирована. "
                "Вызовите ModelContainer.set_empathy() или init_models_from_config()."
            )
        return self._empathy

    @property
    def semantic(self) -> SemanticMetric:
        if self._semantic is None:
            raise RuntimeError(
                "Семантическая модель не инициализирована. "
                "Вызовите ModelContainer.set_semantic() или init_models_from_config()."
            )
        return self._semantic

    # ------------------------------------------------------------------
    # Удобные методы вычисления (делегируют активным моделям)
    # ------------------------------------------------------------------

    def compute_ptox(self, text: str) -> float:
        return self.toxicity.predict(text)

    def compute_pemp(self, text: str) -> float:
        return self.empathy.predict(text)

    def compute_ssem_full(self, text: str, top_k: int = 3) -> dict:
        return self.semantic.predict(text, top_k=top_k)


# ---------------------------------------------------------------------------
# Глобальный синглтон контейнера
# ---------------------------------------------------------------------------

_container: Optional[ModelContainer] = None


def get_model_container() -> ModelContainer:
    """Возвращает глобальный ModelContainer. Создаёт его при первом вызове."""
    global _container
    if _container is None:
        _container = ModelContainer()
    return _container
