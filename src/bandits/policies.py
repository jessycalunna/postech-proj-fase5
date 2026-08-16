"""
Políticas de decisão (os "algoritmos" que escolhem qual oferta apresentar).

Duas políticas:

1. `DeterministicBaselinePolicy` — regra fixa: sempre a mesma oferta. É o PISO
   de comparação. Não aprende, não se adapta. Serve para provar que o bandit
   aprende algo que uma regra simples não captura.

2. `ThompsonSamplingPolicy` — abordagem bayesiana. Cada braço mantém uma
   distribuição Beta sobre sua taxa de conversão. A cada round, sorteia uma
   taxa plausível de cada braço e escolhe o melhor. Explora naturalmente onde
   há mais incerteza — sem parâmetro de exploração para ajustar.

Ambas expõem a mesma interface:
- `select_action(t) -> (action, explored)`
- `update(action, conversion) -> None`

Cold-start (partida a frio): o Thompson começa com prior Beta(1,1) (uniforme)
em todos os braços, o que impede que a política "trave" num braço cedo demais,
porque nenhum braço parte com vantagem artificial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RoundOutcome:
    """Resultado de um round da simulação (uma decisão + seu desfecho)."""

    action: int
    conversion: int
    reward: float
    expected_reward: float
    optimal_expected_reward: float
    explored: int


class DeterministicBaselinePolicy:
    """Regra fixa: sempre apresenta a mesma oferta (`fixed_arm`).

    É o baseline determinístico exigido pelo enunciado. Não usa contexto e
    ignora todo feedback (`update` não faz nada).
    """

    name = "Baseline Determinístico"

    def __init__(self, fixed_arm: int, n_arms: int):
        self.fixed_arm = fixed_arm
        self.n_arms = n_arms

    def select_action(self, t: int) -> Tuple[int, int]:
        return self.fixed_arm, 0  # explored=0: nunca explora

    def update(self, action: int, conversion: int) -> None:
        return  # baseline não aprende


class ThompsonSamplingPolicy:
    """Thompson Sampling com posterior Beta por braço (conversão Bernoulli).

    Cada braço tem parâmetros (alpha, beta) de uma distribuição Beta que
    representa nossa crença sobre sua taxa de conversão. `alpha` conta sucessos
    (+1), `beta` conta fracassos (+1). A cada round, sorteamos uma taxa
    plausível de cada braço e escolhemos o de maior recompensa esperada
    amostrada (taxa sorteada × margem).
    """

    name = "Thompson Sampling"

    def __init__(self, n_arms: int, margins: np.ndarray, seed: int = 42):
        self.n_arms = n_arms
        self.margins = margins
        self.rng = np.random.default_rng(seed)
        # Cold-start: prior Beta(1,1) = distribuição uniforme em [0,1]. Nenhum
        # braço parte na frente; a incerteza inicial é máxima e igual para todos.
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)

    def posterior_mean(self) -> np.ndarray:
        """Taxa de conversão esperada por braço (média da Beta)."""
        return self.alpha / (self.alpha + self.beta)

    def select_action(self, t: int) -> Tuple[int, int]:
        sampled_rate = self.rng.beta(self.alpha, self.beta)
        sampled_reward = sampled_rate * self.margins
        action = int(np.argmax(sampled_reward))

        # "explored" = escolheu algo diferente do que a média posterior indicaria.
        greedy = int(np.argmax(self.posterior_mean() * self.margins))
        explored = int(action != greedy)
        return action, explored

    def update(self, action: int, conversion: int) -> None:
        self.alpha[action] += conversion
        self.beta[action] += 1 - conversion


def build_policy(
    kind: str,
    n_arms: int,
    margins: np.ndarray,
    fixed_arm: int = 0,
    seed: int = 42,
):
    """Fábrica de políticas por nome — usada por notebooks e API.

    `kind` ∈ {"baseline", "thompson"}.
    """
    kind = kind.lower()
    if kind in ("baseline", "fixed", "deterministic"):
        return DeterministicBaselinePolicy(fixed_arm=fixed_arm, n_arms=n_arms)
    if kind in ("thompson", "ts", "thompson_sampling"):
        return ThompsonSamplingPolicy(n_arms=n_arms, margins=margins, seed=seed)
    raise ValueError(f"Política desconhecida: {kind!r}")
