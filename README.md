# Plataforma de Experimentação Adaptativa (Multi-Armed Bandits)

FIAP, Pós Tech 7MLET, Fase 5 (Datathon)

Decidir, para cada cliente elegível, qual oferta apresentar, equilibrando
exploração (descobrir o que funciona) e explotação (usar o que já funciona), por
meio de um multi-armed bandit servido de ponta a ponta no Databricks.

A recompensa usada nas comparações é sintética e calibrada na taxa real de
conversão da base. Não representa uplift de produção; isso é declarado nos
notebooks e neste documento.

## 1. Visão do problema

Uma instituição financeira digital precisa escolher a próxima melhor oferta para
cada cliente. Regras fixas não se adaptam e testes A/B longos desperdiçam
tráfego. Um bandit aprende enquanto decide:

- Cada braço é uma oferta: No Incentive, Fee Discount, Cashback ou Premium Bundle.
- Cada oferta tem uma margem (quanto o banco ganha se houver conversão) e uma
  probabilidade de conversão desconhecida, a ser aprendida.
- O objetivo é maximizar a recompensa esperada, definida como
  probabilidade de conversão multiplicada pela margem da oferta.

## 2. Base de dados

Bank Marketing (Kaggle): https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

Arquivo utilizado: `bank-additional-full.csv` (separador `;`, cerca de 41 mil
linhas). A coluna alvo `y` indica se o cliente contratou um depósito a prazo.

A coluna `duration` (duração da ligação) é descartada por representar vazamento
temporal: seu valor só é conhecido após o contato, e a decisão da oferta ocorre
antes. Após tratamento e remoção de duplicatas, a tabela `bank_clean` tem 39.404
registros, com taxa de conversão geral de 11,67 por cento, valor usado como
âncora da calibração.

## 3. Algoritmos

| Papel | Algoritmo | Arquivo |
|:--|:--|:--|
| Baseline determinístico (piso de comparação) | Regra fixa: sempre a mesma oferta | `src/bandits/policies.py` |
| Algoritmo adaptativo | Thompson Sampling (posterior Beta) | `src/bandits/policies.py` |

Como algoritmo adaptativo, adotamos o Thompson Sampling em vez do Epsilon-Greedy:
ele explora proporcionalmente à incerteza, sem um parâmetro de exploração a
calibrar, e é o algoritmo cujo posterior alimenta o recomendador servido pela
API. O cold-start é tratado pelo prior uniforme Beta(1,1), de modo que nenhum
braço parte com vantagem.

## 4. Resultados

Simulação com 8.000 rounds e semente 42, verificada no Databricks:

| Política | Recompensa total | Taxa de conversão | Arrependimento | Ganho vs baseline |
|:--|--:|--:|--:|--:|
| Thompson Sampling | 1774,1 | 31,0% | 312,3 | +12,6% |
| Baseline determinístico | 1576,0 | 19,7% | 481,7 | referência |

O Thompson Sampling supera o baseline em recompensa e conversão, com menor
arrependimento. As taxas de conversão aprendidas por oferta (posterior médio)
foram No Incentive 0,168, Fee Discount 0,190, Cashback 0,329 e Premium Bundle
0,276: o algoritmo identificou o Cashback como a melhor oferta média.

## 5. Casos de teste (Golden Set)

Cinco clientes representativos submetidos ao recomendador, com as saídas reais do
serviço:

| Perfil do cliente | Oferta recomendada | A decisão fez sentido? |
|:--|:--|:--|
| Engajado (sucesso na campanha anterior, 3 contatos) | Cashback | Sim. Cliente engajado; Cashback maximiza probabilidade vezes margem. |
| Novo, sem histórico (poutcome nonexistent, 0 contatos) | No Incentive | Sim. Sem engajamento confirmado, fica restrito às ofertas de menor custo; entre elas, No Incentive tem a maior recompensa esperada. |
| Conservador idoso (60 anos, falha anterior) | No Incentive | Sim. Perfil conservador e sem engajamento leva à oferta de menor custo. |
| Inadimplente e engajado (default sim, sucesso anterior) | Cashback rebaixado para No Incentive | Sim. A regra de conduta impede incentivo a inadimplente, mesmo quando a métrica indicaria Cashback. |
| Alto valor engajado (4 contatos) | Cashback | Sim. Cliente engajado; Cashback é a oferta de maior recompensa esperada. |

O conjunto completo de avaliação tem 23 casos (típicos, de borda, por segmento e
adversariais) em `data/golden_set/evaluation_cases.jsonl`, validados no Notebook
03. A tabela acima é um recorte de cinco exemplos representativos.

## 6. Estrutura do repositório

```
fiap-fase5/
├── README.md
├── requirements.txt
├── pyproject.toml
├── notebooks/
│   ├── 01_eda_e_tratamento.py          # EDA e tratamento de dados
│   ├── 02_bandits_baseline_mlflow.py   # simulação, baseline, Thompson, MLflow
│   └── 03_avaliacao_golden_set_rag.py  # avaliação, golden set e assistente
├── src/bandits/
│   ├── preprocess.py       # leitura e controle de vazamento
│   ├── synthetic.py        # ambiente de simulação (contexto, mundo, log)
│   ├── policies.py         # baseline e Thompson Sampling
│   ├── simulator.py        # simulação online e métricas
│   ├── recommender.py      # cliente para oferta
│   └── assistant.py        # regras de conduta e explicação da decisão
├── service/
│   └── app.py              # API FastAPI (cliente para oferta)
├── scripts/
│   └── build_golden_set.py # gera o golden set
└── data/
    └── golden_set/
        └── evaluation_cases.jsonl
```

## 7. Como executar

### No Databricks

Ambiente: `jessyca_demos.datathon`. O CSV está no Volume
`/Volumes/jessyca_demos/datathon/raw/`.

Execute, em ordem, os notebooks em `notebooks/`:

1. `01_eda_e_tratamento`: cria a tabela `bank_clean`.
2. `02_bandits_baseline_mlflow`: simula, compara baseline e Thompson, registra
   no MLflow e salva as taxas aprendidas e as tabelas de resultado.
3. `03_avaliacao_golden_set_rag`: avaliação e golden set.

### Localmente (API)

```bash
uv venv && uv pip install -r requirements.txt
uv run uvicorn service.app:app --reload
# abra http://127.0.0.1:8000/docs
```

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"age":34,"job":"management","poutcome":"success","previous":3,"default":"no"}'
```

A resposta traz a oferta recomendada, a oferta permitida após as regras de
conduta, a recompensa esperada e uma justificativa da decisão.

## 8. Ciclo de vida MLOps

Os parâmetros e as métricas da simulação são registrados no MLflow (experimento
`mlflow_bandits`): parâmetros globais da simulação (semente, número de rounds,
distribuição do atraso, taxa base real) e, por política, recompensa total,
arrependimento, taxa de conversão e ganho sobre o baseline. O experimento fica
versionado e comparável.

## 9. Arquitetura-alvo em nuvem

O projeto roda na Databricks, que concentra os serviços necessários: Unity
Catalog para governança e versionamento dos dados, compute serverless ou cluster
para o processamento, MLflow gerenciado para o versionamento dos experimentos e
Databricks Apps para publicar o serviço.

A Databricks está disponível em AWS, Azure e GCP, com os mesmos serviços em
qualquer uma delas. Portanto, esta arquitetura pode ser colocada em produção em
qualquer uma das três nuvens sem alteração do desenho: muda apenas a
infraestrutura subjacente (armazenamento de objetos e compute), enquanto Unity
Catalog, MLflow e Databricks Apps permanecem os mesmos.

## 10. Limitações

- A recompensa é sintética e calibrada na taxa real de conversão; não é uplift de
  produção.
- A conversão é modelada como Bernoulli, sem valor de longo prazo ou churn.
- As margens das ofertas são escalares, sem restrições completas de orçamento.
- Extensões naturais: bandit contextual (LinUCB ou Thompson contextual),
  restrições de orçamento por segmento e avaliação off-policy antes de produção.
