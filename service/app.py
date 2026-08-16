
"""
API demonstrável (Etapa 5) — recebe os dados de um cliente e retorna a oferta
recomendada + a explicação do assistente.

Como rodar localmente:
    uv run uvicorn service.app:app --reload
    # depois abra http://127.0.0.1:8000/docs

Endpoints:
- GET  /health        -> verificação simples
- GET  /offers        -> catálogo de ofertas (braços)
- POST /recommend     -> recebe o contexto do cliente, devolve oferta + explicação

A API carrega as taxas de conversão aprendidas pelo bandit (arquivo
`artifacts/arm_conversion_rates.json`, gerado pelo notebook 02). Se o arquivo
não existir, usa taxas neutras — a API continua funcionando para a demonstração.

A explicação usa a Foundation Model API (Claude) quando as credenciais
Databricks estão disponíveis no ambiente; caso contrário, usa o fallback
determinístico. Assim a API roda em qualquer lugar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.bandits.assistant import (
    OfferAssistant,
    PolicyRAG,
    build_fmapi_caller,
    load_policy_chunks,
)
from src.bandits.recommender import BanditRecommender
from src.bandits.synthetic import OFFER_CATALOG

ROOT = Path(__file__).resolve().parents[1]
POLICIES_DIR = ROOT / "data" / "policies"
RATES_FILE = ROOT / "artifacts" / "arm_conversion_rates.json"

app = FastAPI(
    title="Plataforma de Experimentação Adaptativa — API de Ofertas",
    description="Recebe o contexto de um cliente e recomenda a próxima melhor oferta (multi-armed bandit).",
    version="1.0.0",
)


# --- Modelos de entrada/saída (contrato da API) ---
class ClientContext(BaseModel):
    age: int = Field(..., examples=[35])
    job: str = Field("admin.", examples=["management"])
    marital: str = Field("single", examples=["married"])
    education: str = Field("university.degree", examples=["high.school"])
    default: str = Field("no", examples=["no"])
    housing: str = Field("yes", examples=["yes"])
    loan: str = Field("no", examples=["no"])
    poutcome: str = Field("nonexistent", examples=["success"])
    previous: int = Field(0, examples=[2])
    # Campos opcionais de governança (orçamento etc.)
    orcamento: Optional[str] = Field(None, examples=["ok"])


class RecommendResponse(BaseModel):
    recommended_offer: str
    allowed_offer: str
    needs_human_review: bool
    triggered_rules: list[str]
    expected_conversion: float
    expected_reward: float
    scores: Dict[str, Optional[float]]  # ofertas não elegíveis ao perfil vêm como null
    rationale: str
    retrieved_policies: list[str]
    explanation_source: str


# --- Inicialização (uma vez, no startup) ---
def _load_rates() -> Optional[np.ndarray]:
    if RATES_FILE.exists():
        data = json.loads(RATES_FILE.read_text(encoding="utf-8"))
        return np.array(data["rates"], dtype=float)
    return None


_rag = PolicyRAG(load_policy_chunks(POLICIES_DIR))
_assistant = OfferAssistant(_rag, fmapi_caller=build_fmapi_caller())
_recommender = BanditRecommender(arm_conversion_rates=_load_rates())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "rag_mode": _rag.mode,
        "fmapi_enabled": _assistant.call_llm is not None,
        "learned_rates_loaded": RATES_FILE.exists(),
    }


@app.get("/offers")
def offers() -> list[dict]:
    return OFFER_CATALOG.to_dict(orient="records")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(ctx: ClientContext) -> RecommendResponse:
    context = ctx.model_dump(exclude_none=True)
    rec = _recommender.recommend(context)
    exp = _assistant.explain(context, rec.offer)
    return RecommendResponse(
        recommended_offer=rec.offer,
        allowed_offer=exp.allowed_offer,
        needs_human_review=exp.needs_review,
        triggered_rules=exp.triggered_rules,
        expected_conversion=rec.expected_conversion,
        expected_reward=rec.expected_reward,
        scores=rec.scores,
        rationale=exp.rationale,
        retrieved_policies=exp.retrieved_docs,
        explanation_source=exp.source,
    )
