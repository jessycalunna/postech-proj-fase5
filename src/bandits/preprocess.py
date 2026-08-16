"""
Leitura e tratamento da base Bank Marketing (Kaggle).

Base: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
Arquivo usado: `bank-additional-full.csv` (separador `;`).

O ponto mais importante deste módulo é o **controle de vazamento temporal
(data leakage)**: a coluna `duration` (duração da última ligação em segundos)
só é conhecida DEPOIS que o contato termina. No momento em que precisamos
decidir qual oferta apresentar, ainda não sabemos quanto tempo a ligação vai
durar. Usar `duration` "vazaria" informação do futuro e inflaria o modelo de
forma irreal. Por isso ela é descartada — como o próprio enunciado exige.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Colunas que representam vazamento temporal ou identificadores sem valor
# preditivo legítimo no momento da decisão.
LEAKAGE_COLUMNS: List[str] = [
    "duration",  # só conhecida após o contato terminar -> vazamento clássico
]

# Coluna alvo (assinou depósito a prazo? yes/no) usada como ÂNCORA FACTUAL
# para calibrar a recompensa sintética (não como recompensa crua de um braço).
TARGET_COLUMN = "y"

# Features categóricas e numéricas do cliente/campanha (o contexto x da decisão).
CATEGORICAL_FEATURES: List[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_FEATURES: List[str] = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


def load_bank_marketing(csv_path: str | Path) -> pd.DataFrame:
    """Lê o CSV da base Bank Marketing (separador `;`).

    Retorna o DataFrame cru, sem tratamento — útil para a EDA inicial, onde
    queremos inspecionar inclusive as colunas que depois serão removidas.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado em {csv_path}. Baixe a base do Kaggle "
            "(henriqueyamahata/bank-marketing) e ajuste o caminho. Veja o README."
        )
    df = pd.read_csv(csv_path, sep=";")
    return df


def normalize_target(series: pd.Series) -> pd.Series:
    """Converte o alvo textual (`yes`/`no`) em 0/1.

    `yes` -> 1 (cliente assinou o depósito), `no` -> 0.
    """
    mapping = {"yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0}
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(0)
        .astype(int)
    )


def prepare_dataset(
    df: pd.DataFrame,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Aplica o tratamento mínimo para deixar as features + alvo prontas.

    Passos (todos simples e explicáveis para iniciante):
    1. Remove colunas de vazamento temporal (`duration`).
    2. Normaliza o alvo `y` para 0/1.
    3. Garante tipos numéricos nas colunas numéricas.
    4. Remove linhas duplicadas.

    Não fazemos imputação sofisticada nem one-hot aqui: isso fica a cargo do
    `ColumnTransformer` na hora de montar o contexto (ver `synthetic.py`), para
    manter cada etapa com responsabilidade única.
    """
    df = df.copy()

    if drop_leakage:
        present = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        df = df.drop(columns=present)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = normalize_target(df[TARGET_COLUMN])

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Retorna (categóricas, numéricas) efetivamente presentes no DataFrame."""
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    nums = [c for c in NUMERIC_FEATURES if c in df.columns]
    return cats, nums


def sample_for_simulation(
    df: pd.DataFrame,
    n_rounds: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Amostra `n_rounds` clientes para servir de contexto na simulação.

    Cada linha amostrada vira um "round" (uma decisão) no simulador. Amostrar
    mantém a simulação rápida e reprodutível. Se `n_rounds` for None, usa a
    base inteira.
    """
    if n_rounds is None or n_rounds >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n_rounds, replace=False)
    return df.iloc[idx].reset_index(drop=True)
