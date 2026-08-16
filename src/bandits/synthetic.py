"""
Camada sintética de experimentação (o "simulador" onde bandit e baseline competem).

Por que precisamos dela? A base Kaggle só registra a resposta a UMA abordagem
(assinou ou não o depósito). Um multi-armed bandit precisa de VÁRIOS braços
(ofertas) para escolher entre eles. Como não temos a resposta real do cliente
para cada oferta possível, construímos um ambiente sintético e reprodutível:

1. **Contexto (x):** vetor denso derivado das features reais do cliente.
2. **Catálogo de ofertas (braços):** No Incentive, Fee Discount, Cashback,
   Premium Bundle — cada uma com uma margem de negócio diferente.
3. **Probabilidade latente de conversão por braço:** uma função do contexto,
   desconhecida das políticas. É o que elas tentam descobrir.
4. **Interações logadas + recompensas atrasadas:** o histórico "observado".

Calibração com o `y` real (âncora factual): a taxa de conversão do braço de
referência ("No Incentive") é ancorada na taxa real de assinatura da base. As
demais ofertas partem dessa âncora e recebem um empurrão (uplift) — ofertas
mais generosas convertem mais, mas custam margem. Assim a comparação é justa
por construção: baseline e bandit rodam no MESMO ambiente e MESMA semente.

IMPORTANTE: a recompensa aqui é sintética e calibrada, NÃO é uplift de produção.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_onehot() -> OneHotEncoder:
    """Cria um OneHotEncoder esparso compatível com versões do scikit-learn.

    O parâmetro foi renomeado de `sparse` para `sparse_output` na versão 1.2.
    Como o runtime do Databricks pode ter uma versão mais antiga, tentamos o
    nome novo e caímos no antigo se necessário.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# ---------------------------------------------------------------------------
# Catálogo de ofertas (os braços do bandit)
# ---------------------------------------------------------------------------
# Cada oferta tem:
# - `arm`: índice do braço.
# - `offer_name`: nome de negócio.
# - `margin`: margem relativa da oferta (quanto o banco ganha se converter).
#   Ofertas mais generosas convertem mais, mas têm margem menor por conversão.
# - `uplift`: quanto essa oferta aumenta (em logito) a chance de conversão em
#   relação ao braço de referência. É a hipótese de calibração sintética.
OFFER_CATALOG = pd.DataFrame(
    [
        {"arm": 0, "offer_name": "No Incentive", "margin": 1.00, "uplift": 0.00},
        {"arm": 1, "offer_name": "Fee Discount", "margin": 0.85, "uplift": 0.45},
        {"arm": 2, "offer_name": "Cashback", "margin": 0.70, "uplift": 0.80},
        {"arm": 3, "offer_name": "Premium Bundle", "margin": 0.55, "uplift": 1.10},
    ]
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Função logística: transforma qualquer número real em uma probabilidade (0..1)."""
    return 1.0 / (1.0 + np.exp(-z))


def build_context_matrix(
    df: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    n_components: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, Pipeline]:
    """Transforma as features do cliente em um vetor de contexto denso e compacto.

    Passos:
    1. One-hot nas categóricas + padronização das numéricas (ColumnTransformer).
    2. Redução de dimensionalidade com TruncatedSVD para um vetor pequeno
       (`n_components`), o que deixa a simulação leve e estável.

    Retorna a matriz de contextos (n_clientes x n_components) e o pipeline
    ajustado (para reutilizar na API e nos notebooks).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _make_onehot(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    X_sparse = preprocessor.fit_transform(df)
    max_comp = max(2, min(n_components, X_sparse.shape[1] - 1))
    reducer = TruncatedSVD(n_components=max_comp, random_state=seed)
    X_dense = reducer.fit_transform(X_sparse)

    pipeline = Pipeline([("preprocessor", preprocessor), ("reducer", reducer)])
    return X_dense, pipeline


def build_true_reward_model(
    contexts: np.ndarray,
    target: np.ndarray,
    offer_catalog: pd.DataFrame = OFFER_CATALOG,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrói o "mundo verdadeiro" (desconhecido das políticas).

    Retorna:
    - `all_probs`   (n x k): probabilidade latente de conversão de cada braço.
    - `margins`     (k,)   : margem de cada oferta.
    - `true_weights`(k x d): pesos latentes (guardados só para inspeção).

    Calibração (âncora factual com o `y` real):
    - Calculamos a taxa real de conversão da base (`base_rate`, ex.: ~11%).
    - Convertemos essa taxa para logito e usamos como bias do braço de
      referência (arm 0, "No Incentive"). Ou seja, sem incentivo, a chance de
      conversão parte da realidade observada.
    - Cada braço recebe um `uplift` (definido no catálogo) somado ao bias.
    - Uma pequena dependência do contexto (pesos aleatórios com semente fixa)
      faz a oferta ótima variar de cliente para cliente — é isso que dá ao
      bandit contextual algo para aprender.
    """
    rng = np.random.default_rng(seed)
    n, d = contexts.shape
    k = len(offer_catalog)

    margins = offer_catalog["margin"].to_numpy(dtype=float)
    uplifts = offer_catalog["uplift"].to_numpy(dtype=float)

    # Âncora factual: taxa real de conversão -> logito -> bias do braço 0.
    base_rate = float(np.clip(np.mean(target), 1e-3, 1 - 1e-3))
    base_logit = float(np.log(base_rate / (1.0 - base_rate)))

    # Bias por braço = âncora + uplift da oferta.
    true_bias = base_logit + uplifts

    # Dependência do contexto: pesos pequenos para não dominar a âncora.
    true_weights = rng.normal(loc=0.0, scale=0.35, size=(k, d))

    all_logits = contexts @ true_weights.T + true_bias  # (n x k)
    all_probs = sigmoid(all_logits)
    return all_probs, margins, true_weights


def generate_logged_interactions(
    all_probs: np.ndarray,
    margins: np.ndarray,
    epsilon: float = 0.35,
    delay_p: float = 0.35,
    max_delay: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera o histórico de interações "observado" (log de impressões).

    Simula uma política de logging epsilon-gulosa (parte gulosa pela recompensa
    esperada, parte aleatória para exploração). Cada linha é uma impressão:
    "apresentei a oferta `action` ao cliente do round `round`".

    Recompensas atrasadas: cada impressão recebe `delay_steps` (amostrado de uma
    distribuição geométrica, truncada em `max_delay`). Isso modela que o feedback
    de conversão só "amadurece" alguns rounds depois — como no mundo real, em que
    o cliente não responde instantaneamente.
    """
    rng = np.random.default_rng(seed)
    n, k = all_probs.shape
    expected_reward = all_probs * margins  # (n x k)

    actions = np.zeros(n, dtype=int)
    chosen_prob = np.zeros(n, dtype=float)
    conversion = np.zeros(n, dtype=int)
    reward = np.zeros(n, dtype=float)
    delay_steps = np.zeros(n, dtype=int)

    for t in range(n):
        greedy_arm = int(np.argmax(expected_reward[t]))
        if rng.random() < epsilon:
            a = int(rng.integers(0, k))
        else:
            a = greedy_arm

        p = all_probs[t, a]
        conv = int(rng.random() < p)
        actions[t] = a
        chosen_prob[t] = p
        conversion[t] = conv
        reward[t] = conv * margins[a]
        delay_steps[t] = int(min(rng.geometric(delay_p) - 1, max_delay))

    return pd.DataFrame(
        {
            "round": np.arange(n),
            "action": actions,
            "true_p": chosen_prob,
            "conversion": conversion,
            "reward": reward,
            "delay_steps": delay_steps,
        }
    )
