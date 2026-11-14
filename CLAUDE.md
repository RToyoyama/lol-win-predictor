# CLAUDE.md — LoL Win Predictor (Projeto Final EBAC · Cientista de Dados)

## 1. Contexto do projeto

**Stakeholder (fictício):** Tech Lead de dados da Riot Games.
**Pergunta de negócio:** *"Aos 10 minutos de partida, qual a probabilidade de o lado azul vencer?"*
**Usos propostos:** (1) barra de win-probability para transmissões; (2) indicador de "jogo decidido" (contexto do surrender aos 15); (3) análise de equilíbrio entre lados e valor de objetivos.
**Entrega:** repositório documentado + notebooks narrativos + API + app demo + vídeo de apresentação (definido depois).
**Critérios EBAC:** originalidade, rigor analítico e clareza na comunicação.

O autor joga LoL e entende o jogo — use vocabulário do jogo quando ajudar (gold diff, arauto, first blood, visão), sem explicar o básico do jogo.

## 2. Dados — fatos já verificados (não redescobrir, apenas confirmar na EDA)

- Arquivo: `data/raw/Base_M43_Pratique_LOL_RANKED_WIN.csv` — 9.879 partidas ranqueadas (elo alto), 40 colunas.
- Cada linha é um **snapshot aos 10 minutos**. Alvo: `blueWins` (0/1).
- Sem nulos, sem `gameId` duplicado, classes balanceadas (50,1% / 49,9%).
- **Não há dados de pré-jogo** (campeões, picks/bans, MMR). **Não usar dados de pós-jogo** (vazamento).
- Redundâncias determinísticas (100% das linhas):
  - `blueKills == redDeaths` e `redKills == blueDeaths`
  - `blueGoldDiff == -redGoldDiff == blueTotalGold - redTotalGold`
  - `blueExperienceDiff == -redExperienceDiff`
  - `GoldPerMin == TotalGold / 10`, `CSPerMin == TotalMinionsKilled / 10`
  - `EliteMonsters == Dragons + Heralds`
  - `blueFirstBlood + redFirstBlood == 1`
  - Dragões: no máximo 1 no total aos 10 min (`blueDragons + redDragons <= 1`); arautos ≤ 1 por lado.
- Outliers: `WardsPlaced` tem mediana 16 e máximo 250–276; ~1% das linhas > 100. Investigar na EDA e decidir tratamento (cap por percentil **ajustado só no treino**, dentro do pipeline).
- Benchmark: regressão logística só com `blueGoldDiff` → **ROC-AUC ≈ 0,80** (CV 5-fold). Teto realista do projeto: **~0,80–0,82**.
  **Se qualquer modelo passar de 0,85 de AUC, PARE e investigue vazamento.**

## 3. Stack e padrões

- Python 3.11 · gerenciador **uv** (`pyproject.toml`) · `src/` layout (pacote `lol_win`)
- pandas, pyarrow, **pandera**, scikit-learn, lightgbm, optuna, shap, statsmodels, scipy
- matplotlib/seaborn (notebooks), **mlflow** (local, `mlruns/`), joblib
- **fastapi** + pydantic v2 + uvicorn · **streamlit**
- pytest, ruff (lint + format), pre-commit · Docker + docker-compose (api + app) · GitHub Actions
- **Fora de escopo (decisão consciente):** Spark, dbt, DuckDB, Airflow, Riot API, monitoramento de drift ativo (vira seção "como eu monitoraria em produção" no README).

**Código:**
- snake_case · **type hints obrigatórios** · **docstrings padrão Google** em funções/classes públicas
- Sem comentários óbvios · funções pequenas e testáveis · caminhos e constantes em `config.py`
- `RANDOM_STATE = 42` em todo lugar
- Notebooks **importam de `src/lol_win`**; nenhuma lógica de transformação duplicada em notebook
- Figuras salvas em `reports/figures/` com nomes descritivos

## 4. Arquitetura de dados (pandas + Parquet)

```
raw    → CSV original, imutável
bronze → tipado, colunas em snake_case, validado com Pandera (schema + checks de consistência da seção 2)
silver → colunas redundantes removidas, features base limpas
gold   → features de modelagem (diffs) + alvo; split treino/teste fixo
```

**Split:** holdout de teste **20% estratificado**, criado uma única vez na camada gold e salvo (`gold/train.parquet`, `gold/test.parquet`). O teste **só é usado na avaliação final** (Etapa 6). Toda seleção/tuning usa CV estratificada 5-fold no treino.

**Features (ponto de partida, refinar na Etapa 4):** diferenças azul − vermelho de kills, assists, wards placed, wards destroyed, dragons, heralds, towers, minions (CS), jungle CS, gold, xp, avg level; `blue_first_blood`. Comparar conjunto "só diffs" vs "diffs + absolutos".

**Regra anti training-serving skew:** a mesma função `build_features()` é usada no treino, na API e no app. A API recebe estatísticas brutas dos dois times aos 10 min e calcula as features internamente.

## 5. Estrutura de pastas

```
lol-win-predictor/
├── CLAUDE.md
├── README.md
├── Makefile
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── data/
│   ├── raw/            # CSV versionado (público e pequeno)
│   ├── bronze/         # gitignored
│   ├── silver/         # gitignored
│   └── gold/           # gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hipoteses_de_jogo.ipynb
│   └── 03_interpretacao_e_decisao.ipynb
├── src/lol_win/
│   ├── __init__.py
│   ├── config.py
│   ├── data/{ingest.py, schemas.py, clean.py}
│   ├── features/build.py
│   ├── models/{baselines.py, train.py, tune.py, calibrate.py, evaluate.py, predict.py}
│   ├── api/{main.py, schemas.py}
│   └── app/streamlit_app.py
├── models/             # artefato final exportado (gitignored)
├── mlruns/             # gitignored
├── reports/
│   ├── figures/
│   ├── metrics/        # JSON com métricas finais
│   └── model_card.md
├── docs/
│   └── roteiro_video.md
└── tests/
```

## 6. Etapas do projeto

Cada etapa termina com: código funcionando + testes passando + commit. Etapas grandes viram parte A/B, cada uma com seu commit.

**Pré-projeto (antes da Etapa 0):** estrutura de pastas, checklist de pré-requisitos (ferramentas + versões), `.gitignore` contextual (Python, notebooks, dados intermediários, mlruns, modelos, .env, IDEs).

**Etapa 0 — Setup** [pode ser trivial]
`pyproject.toml` com uv, Makefile (`setup, data, features, train, tune, evaluate, api, app, test, lint, docker-up`), pre-commit (ruff, ruff-format, nbstripout opcional *não* — manter outputs dos notebooks), CI mínimo (ruff + pytest), `config.py`.

**Etapa 1 — Ingestão e contratos de dados**
`ingest.py` (raw → bronze), `schemas.py` com Pandera: tipos, ranges e **todas as regras de consistência da seção 2**. Testes que provam que o schema rejeita dados inválidos.

**Etapa 2 — Notebook 01: EDA**
Estrutura, resumo estatístico, distribuições, outliers de wards (decisão documentada), matriz de correlação, multicolinearidade (VIF), relação de cada variável com `blue_wins`, taxa de vitória do lado azul. Termina com seção "Decisões tomadas" que alimenta a Etapa 4.

**Etapa 3 — Notebook 02: Hipóteses de jogo** (diferencial do projeto)
Cada hipótese: H0/H1, teste, tamanho de efeito, IC, conclusão em linguagem de jogador.
- **Vantagem de lado:** teste binomial de `blue_wins` vs 0,5.
- **Quanto vale um dragão / arauto / torre em gold?** Logit (statsmodels) com `gold_diff` + objetivo; valor em gold = β_objetivo / β_gold_diff, IC por bootstrap. Ressalva: é o valor **além** do gold que o objetivo já gerou.
- **First blood decide ou só acompanha quem já está na frente?** Qui-quadrado bruto vs logit controlando `gold_diff` — mostrar o efeito encolhendo (confusão).
- **Visão importa no early?** Mesmo desenho com `wards_placed_diff` e `wards_destroyed_diff`.

**Etapa 4 — Feature engineering**
Bronze → silver → gold, `build_features()`, split fixo. Testes unitários das features (incluindo simetria: inverter lados ⇒ diffs com sinal trocado).

**Etapa 5 — Baselines e modelos**
Baselines: `DummyClassifier` e regra "quem tem mais gold vence". Modelos: Regressão Logística (StandardScaler), Random Forest, LightGBM — todos como `sklearn.Pipeline`. CV 5-fold estratificada. Métricas: **ROC-AUC (principal)**, log loss, Brier, acurácia, F1. Tudo registrado no MLflow (params, métricas, artefatos, tags do conjunto de features). Tabela comparativa.

**Etapa 6 — Tuning, calibração e avaliação final**
Optuna nos 1–2 melhores (otimizar log loss ou AUC, justificar). Calibração (`CalibratedClassifierCV`, sigmoid vs isotônica) com curva de calibração + Brier. Escolha final justificada (desempenho × simplicidade × calibração). **Só agora** avaliar no teste, com IC por bootstrap. Registrar no MLflow Model Registry, exportar para `models/` e salvar `reports/metrics/final.json`.

**Etapa 7 — Notebook 03: Interpretação e decisão**
SHAP global (importância, beeswarm) e local (waterfall de 2–3 partidas). **Análise das viradas:** partidas em que o modelo deu >80% e o time perdeu — o que elas têm em comum? **Faixas de decisão:** tabela por faixa de probabilidade (ex.: <15%, 15–35%, 35–65%, 65–85%, >85%) com % de partidas na faixa e taxa de acerto → frase de negócio do tipo "aos 10 min dá para cravar o resultado em X% das partidas". Gerar `reports/model_card.md`.

**Etapa 8 — API (FastAPI)**
`GET /health`, `POST /predict` (stats brutas dos dois times aos 10 min → `p_blue_win`, `p_red_win`, `faixa`), `POST /predict/batch`. Validação Pydantic com regras do jogo (valores ≥ 0, dragões somados ≤ 1, arautos ≤ 1 por lado, etc.). Modelo carregado uma vez no startup. Testes com `TestClient`.

**Etapa 9 — App Streamlit**
Simulador "e se?": sliders/inputs para os dois times, barra de win-probability estilo HUD, faixa de decisão, waterfall SHAP do cenário e botões rápidos ("+1 dragão azul", "+1.000 de gold"). Reutiliza `predict.py`.

**Etapa 10 — Docker + CI completo**
Dockerfile multi-stage, `docker-compose.yml` (api + app), CI: lint + testes + smoke test de treino em amostra pequena.

**Etapa 11 — README final + roteiro do vídeo**
README para recrutador: problema, resultado em 3 linhas, principais insights de jogo, arquitetura (diagrama), como rodar (1 comando), decisões e trade-offs, limitações, "como eu monitoraria em produção", próximos passos (Riot API, dados por minuto, modelo de pré-jogo por composição). `docs/roteiro_video.md`: roteiro de 5–7 min em cenas (fala + visual sugerido). O formato do vídeo será decidido depois.

## 7. Protocolo de trabalho (MODO TUTOR)

- Antes de cada etapa: **mini-roadmap** da etapa (o que, por quê, arquivos que serão criados).
- Antes do código: **conceito + porquê + boas práticas + quando usar vs quando NÃO usar** (máx. 10 linhas).
- Nível padrão: **intermediário** (mudar com `NÍVEL: iniciante | intermediário | avançado`).
- Ao fim da etapa: rodar testes/lint, mostrar resultado e propor o commit. **Executar `git commit` só após minha confirmação.**
- **Só avançar para a próxima etapa após minha confirmação.** Etapas marcadas [trivial] podem seguir direto.
- Respostas diretas, sem floreio, sem repetir o que já foi dito. Se algo é trivial, só cite.
- Ao encontrar algo que contradiga este arquivo (ex.: AUC suspeita, regra de consistência quebrada), **pare e reporte** antes de seguir.

**Comandos rápidos:**
- `/e` → pula explicação, entrega só o código
- `/skip` → etapa trivial, executa sem confirmação
- `/readme` → gera README.md consolidado até aqui
- `/log` → mostra todos os commits em ordem cronológica (`git log --oneline --reverse`)

## 8. Formato de commit (Conventional Commits)

```
tipo(escopo)[!]: descrição curta em inglês

Descrição em português.
[breaking change: descrição, se aplicável]
```
Tipos: `feat | fix | refactor | docs | chore | test | ci | perf` · `!` = breaking change.

Exemplo:
```
feat(data): add bronze layer with pandera contracts

Implementa ingestão raw → bronze com validação de tipos, ranges e regras de consistência entre colunas dos dois times.
```