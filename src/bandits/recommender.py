"""
Recomendador de oferta: transforma o contexto de UM cliente na oferta escolhida.

Este módulo conecta as peças para o uso "em produção" (API/serviço):
1. Recebe os atributos do cliente.
2. Usa o pipeline de features (mesmo do treino) para gerar o contexto denso.
3. Aplica a política aprendida para escolher o braço.
4. Devolve a oferta + metadados.

Para uma política contextual completa seria natural um LinUCB; aqui, seguindo o
pedido de simplicidade, o "conhecimento" aprendido pelo Thompson Sampling é
resumido na taxa de conversão estimada por braço (posterior médio), e a decisão
final pondera essa taxa pela margem. A dependência de contexto entra por uma
regra de segmentação leve e transparente (perfil), alinhada às políticas.

O objetivo é ser DEMONSTRÁVEL e explicável, não um sistema de produção.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .synthetic import OFFER_CATALOG

OFFER_NAMES: List[str] = OFFER_CATALOG["offer_name"].tolist()
MARGINS: np.ndarray = OFFER_CATALOG["margin"].to_numpy(dtype=float)


@dataclass
class Recommendation:
    offer: str
    arm: int
    expected_conversion: float
    expected_reward: float
    scores: Dict[str, float]


class BanditRecommender:
    """Recomendador baseado nas taxas de conversão aprendidas por braço.

    `arm_conversion_rates`: taxa estimada por braço (ex.: posterior médio do
    Thompson após a simulação). Se não informado, usa uma estimativa neutra.
    """

    def __init__(
        self,
        arm_conversion_rates: Optional[np.ndarray] = None,
        margins: np.ndarray = MARGINS,
        offer_names: List[str] = OFFER_NAMES,
    ):
        self.margins = margins
        self.offer_names = offer_names
        n = len(offer_names)
        if arm_conversion_rates is None:
            # Estimativa neutra a partir do catálogo (uplift relativo).
            self.rates = np.linspace(0.15, 0.30, n)
        else:
            self.rates = np.asarray(arm_conversion_rates, dtype=float)

    def _is_engaged(self, context: Dict) -> bool:
        """Engajamento = sucesso em campanha anterior OU vários contatos prévios."""
        poutcome = str(context.get("poutcome", "nonexistent")).lower()
        previous = context.get("previous", 0) or 0
        return poutcome == "success" or previous >= 2

    def _eligible_mask(self, context: Dict) -> np.ndarray:
        """Conjunto de ofertas elegíveis por perfil (segmentação transparente).

        Implementa a regra R3 da política comercial como uma **restrição do
        conjunto de escolha** (choice-set restriction), mais robusta e auditável
        do que apenas ponderar taxas:

        - Cliente **sem engajamento** confirmado: elegível apenas às ofertas de
          **menor custo** (No Incentive, Fee Discount) até haver evidência.
        - Cliente **engajado**: todas as ofertas ficam elegíveis; o bandit
          escolhe pela maior recompensa esperada.
        """
        mask = np.ones(len(self.offer_names), dtype=bool)
        if not self._is_engaged(context):
            mask[:] = False
            mask[[0, 1]] = True  # No Incentive, Fee Discount
        return mask

    def recommend(self, context: Dict) -> Recommendation:
        rates = np.clip(self.rates, 0.0, 1.0)
        expected_reward = rates * self.margins

        # Elegibilidade por perfil: ofertas não elegíveis recebem -inf e nunca
        # são escolhidas.
        mask = self._eligible_mask(context)
        masked_reward = np.where(mask, expected_reward, -np.inf)
        arm = int(np.argmax(masked_reward))
        scores = {
            self.offer_names[i]: (round(float(expected_reward[i]), 4) if mask[i] else None)
            for i in range(len(self.offer_names))
        }
        return Recommendation(
            offer=self.offer_names[arm],
            arm=arm,
            expected_conversion=round(float(rates[arm]), 4),
            expected_reward=round(float(expected_reward[arm]), 4),
            scores=scores,
        )
