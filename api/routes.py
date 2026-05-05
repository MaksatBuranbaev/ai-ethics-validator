"""
api/routes.py — HTTP-эндпоинты REST API.

Маршруты тонкие: валидируют входные данные (Pydantic),
делегируют бизнес-логику EvaluationService и возвращают ответ.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.dependencies import get_evaluation_service
from services.evaluation_service import EvaluationService

router = APIRouter()


# ---------------------------------------------------------------------------
# Схемы запросов / ответов
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст для оценки")
    tau: float = Field(0.3, ge=0.0, le=1.0, description="Порог токсичности")


class EvaluateResponse(BaseModel):
    ptox: float
    pemp: float
    ssem: float
    ihum: float
    veto: bool
    tau_tox: float
    top_k_phrases: List[Dict[str, Any]]
    verdict: str


class ParetoRequest(BaseModel):
    candidates: List[str] = Field(..., min_length=1, description="Список текстов-кандидатов")
    tau: float = Field(0.3, ge=0.0, le=1.0)


class ParetoResponse(BaseModel):
    results: List[EvaluateResponse]
    pareto_indices: List[int]


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ExplainResponse(BaseModel):
    tokens: List[str]
    importances: List[float]


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@router.get("/health", summary="Проверка работоспособности сервера")
async def health_check() -> dict:
    return {"status": "ok"}


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Оценить один текст",
)
async def evaluate_text(
    request: EvaluateRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    return service.evaluate(request.text, tau_tox=request.tau)


@router.post(
    "/pareto",
    response_model=ParetoResponse,
    summary="Оценить список текстов и построить Парето-фронт",
)
async def pareto_analysis(
    request: ParetoRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    return service.evaluate_multiple(request.candidates, tau_tox=request.tau)


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="XAI: важность токенов для оценки токсичности",
)
async def explain_toxicity(
    request: ExplainRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> ExplainResponse:
    items = service.explain_toxicity(request.text)
    return ExplainResponse(
        tokens=[item["token"] for item in items],
        importances=[item["importance"] for item in items],
    )
