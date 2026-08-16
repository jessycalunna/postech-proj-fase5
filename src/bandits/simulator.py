"""
Motor de simulação online com recompensas atrasadas + métricas de avaliação.

Como funciona cada round `t`:
1. Amadurecem as recompensas cujo atraso venceu em `t` -> chamamos `update()`.
2. A política escolhe uma oferta para o contexto do round `t`.
3. Sorteamos a conversão a partir da probabilidade latente daquele braço.
4. A recompensa observada é agendada para amadurecer em `t + delay`.

Métricas acumuladas:
- `cum_reward`            : recompensa total acumulada (margem × conversões).
- `instant_regret`        : quanto deixamos de ganhar por não ter escolhido a
                            melhor oferta possível naquele contexto.
- `cum_regret`            : arrependimento acumulado (quanto menor, melhor).
- `cum_conversion_rate`   : taxa de conversão acumulada.
- `cum_exploration_share` : fração de rounds em que a política explorou (deve
                            cair ao longo do tempo à medida que ela aprende).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .policies import RoundOutcome


def run_policy_simulation(
    policy_name: str,
    policy,
    all_probs: np.ndarray,
    margins: np.ndarray,
    delay_p: float = 0.35,
    max_delay: int = 8,
    seed: int = 100,
) -> pd.DataFrame:
    """Roda a simulação online de uma política e devolve o histórico por round.

    `all_probs` e `margins` definem o "mundo verdadeiro" (ver synthetic.py).
    Baseline e bandits DEVEM receber os mesmos `all_probs` e a mesma `seed` para
    que a comparação seja justa.
    """
    rng = np.random.default_rng(seed)
    n, _ = all_probs.shape

    pending: Dict[int, List[Tuple[int, int]]] = {}
    rows: List[RoundOutcome] = []

    for t in range(n):
        # 1) Feedback que amadureceu neste round vira aprendizado.
        for act, conv in pending.pop(t, []):
            policy.update(act, conv)

        # 2) Decisão.
        action, explored = policy.select_action(t)

        # 3) Desfecho (conversão sorteada da probabilidade latente).
        p = all_probs[t, action]
        conversion = int(rng.random() < p)
        reward = conversion * margins[action]

        # 4) Agenda a recompensa para amadurecer com atraso.
        delay = int(min(rng.geometric(delay_p) - 1, max_delay))
        pending.setdefault(t + delay, []).append((action, conversion))

        expected_reward = p * margins[action]
        optimal_expected = float(np.max(all_probs[t] * margins))

        rows.append(
            RoundOutcome(
                action=action,
                conversion=conversion,
                reward=reward,
                expected_reward=expected_reward,
                optimal_expected_reward=optimal_expected,
                explored=explored,
            )
        )

    result = pd.DataFrame([r.__dict__ for r in rows])
    result["policy"] = policy_name
    result["round"] = np.arange(len(result))
    result["instant_regret"] = (
        result["optimal_expected_reward"] - result["expected_reward"]
    )
    result["cum_reward"] = result["reward"].cumsum()
    result["cum_regret"] = result["instant_regret"].cumsum()
    result["cum_conversions"] = result["conversion"].cumsum()
    result["cum_conversion_rate"] = result["cum_conversions"] / (result["round"] + 1)
    result["cum_exploration_share"] = result["explored"].cumsum() / (
        result["round"] + 1
    )
    return result


def summarize_simulations(sim_all: pd.DataFrame) -> pd.DataFrame:
    """Resume as métricas finais por política, ordenado por recompensa total."""
    summary = (
        sim_all.groupby("policy", as_index=False)
        .agg(
            total_reward=("reward", "sum"),
            cumulative_regret=("instant_regret", "sum"),
            conversion_rate=("conversion", "mean"),
            exploration_share=("explored", "mean"),
        )
        .sort_values(by="total_reward", ascending=False)
        .reset_index(drop=True)
    )
    return summary
