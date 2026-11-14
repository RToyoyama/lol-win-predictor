# Glossário do projeto

Termos técnicos usados no LoL Win Predictor, em ordem de aparição no pipeline.
Cada verbete traz **o que é** e **onde aparece no projeto**. Termos que ainda vão aparecer
estão marcados com a etapa (ex.: *Etapa 6*).

---

## 1. Ferramentas e engenharia

| Termo | O que é | No projeto |
|---|---|---|
| **uv** | Gerenciador de pacotes e ambientes Python (substitui pip + venv + pip-tools), muito rápido. | `uv sync` instala tudo; `uv run <cmd>` executa dentro do ambiente. |
| **pyproject.toml** | Arquivo padrão que descreve o projeto Python: nome, dependências e configuração das ferramentas. | Dependências de runtime, grupo `dev`, regras do ruff e do pytest. |
| **Lockfile (`uv.lock`)** | Registro das versões **exatas** de todas as dependências, incluindo as indiretas. | Garante que CI, Docker e outras máquinas instalem o mesmo ambiente (`uv sync --frozen`). |
| **Dependências de runtime vs dev** | Runtime: o que o código precisa para rodar. Dev: o que só o desenvolvedor usa (testes, lint, Jupyter). | A imagem Docker de produção não leva pytest nem Jupyter. |
| **src layout** | Código do pacote dentro de `src/`, separado de testes e notebooks. | Pacote `lol_win`, importado por notebooks, API e app. Evita importar o código "solto" por acidente. |
| **Makefile** | Atalhos para comandos longos (`make test`, `make data`...). | Interface única para rodar cada etapa do pipeline. |
| **ruff** | Linter (aponta erros e más práticas) e formatador de código Python. | `make lint`; também verifica os notebooks. |
| **Lint** | Análise estática do código sem executá-lo: imports não usados, nomes ruins, docstrings faltando. | Regras E, F, I, UP, B, SIM, D, N, PD, NPY no `pyproject.toml`. |
| **pre-commit** | Roda verificações automaticamente antes de cada `git commit`; se algo falhar, o commit é bloqueado. | ruff + checagens de espaço, EOF, YAML/TOML e arquivos grandes. |
| **CI (Integração Contínua)** | Servidor que roda lint e testes a cada push. | GitHub Actions em `.github/workflows/ci.yml`. |
| **pytest** | Framework de testes automatizados. | `make test`; testes em `tests/`. |
| **Fixture** | Dado ou objeto preparado para os testes e reutilizado entre eles. | `valid_bronze` em `test_schemas.py`: duas partidas sintéticas válidas. |
| **Teste parametrizado** | Um único teste executado com várias entradas diferentes. | Corrompemos uma regra por vez (21 casos) e esperamos rejeição. |
| **Type hints** | Anotações de tipo (`def f(x: int) -> str`). Não mudam a execução, mas documentam e permitem checagem. | Obrigatórias em todo o código (CLAUDE.md). |
| **Docstring Google** | Padrão de documentação de funções com seções `Args:`, `Returns:` e `Raises:`. | Toda função pública do pacote. |
| **Conventional Commits** | Padrão de mensagem de commit: `tipo(escopo): descrição`. | `feat(data): ...`, `chore(setup): ...`. |
| **jupytext** | Converte notebooks para scripts `.py` e vice-versa. | O notebook é escrito como `.py`, convertido e executado com **nbconvert**, que salva os outputs no `.ipynb`. |

## 2. Arquitetura de dados

| Termo | O que é | No projeto |
|---|---|---|
| **Arquitetura medallion** (raw/bronze/silver/gold) | Camadas progressivas de refinamento do dado; cada camada só lê a anterior. | **raw:** CSV original, imutável. **bronze:** tipado e validado. **silver:** limpo, sem redundâncias. **gold:** features de modelagem + split. |
| **Parquet** | Formato de arquivo colunar e comprimido que preserva tipos. Leitura muito mais rápida que CSV. | Bronze, silver e gold são `.parquet`. |
| **snake_case** | Convenção de nomes `palavras_minusculas_com_underscore`. | `blueCSPerMin` → `blue_cs_per_min` na ingestão. |
| **Contrato de dados** | Especificação formal do que o dado deve ser: colunas, tipos, ranges e regras. | `BRONZE_SCHEMA` em `schemas.py`. |
| **Pandera** | Biblioteca que valida DataFrames contra um contrato. | 21 regras entre colunas + tipos e ranges por coluna. |
| **`strict=True` / `coerce=False`** | `strict`: coluna extra ou faltando é erro. `coerce=False`: não converte tipos silenciosamente (um float onde deveria haver int é erro). | Os dois estão ligados no contrato bronze. |
| **`lazy=True`** | O Pandera reúne **todas** as violações num relatório, em vez de parar na primeira. | `BRONZE_SCHEMA.validate(df, lazy=True)`. |
| **Redundância determinística** | Coluna que é função exata de outra (`gold_per_min = total_gold / 10`). | 13 colunas removidas no silver (*Etapa 4*). |
| **Vazamento de dados (data leakage)** | Usar no treino informação que não estaria disponível no momento da previsão, ou que vem do conjunto de teste. Infla as métricas artificialmente. | Proibido usar dados pós-10 min. AUC > 0,85 é sinal de alerta. |
| **Training-serving skew** | Quando o modelo recebe features calculadas de um jeito no treino e de outro em produção. | Evitado com uma única `build_features()` usada no treino, na API e no app. |

## 3. Estatística (EDA e hipóteses)

| Termo | O que é | No projeto |
|---|---|---|
| **EDA** | Análise Exploratória de Dados: entender os dados antes de modelar. | Notebook 01. |
| **Percentil (p99 etc.)** | Valor abaixo do qual está X% dos dados. p99 = 99% das partidas estão abaixo dele. | `wards_placed`: p50 = 16, p99 ≈ 105, máx = 276. |
| **Outlier** | Valor muito distante do resto da distribuição. | Wards > 100 em 10 min. |
| **Distribuição multimodal** | Distribuição com vários "picos". Na cauda, costuma indicar subgrupos ou artefato de coleta. | Wards têm picos perto de 40, 85 e 145. |
| **Winsorização (cap)** | Limitar valores extremos a um teto (ex.: tudo acima do p99 vira o p99), sem apagar a linha. | Cap no p99 de `wards_placed` (*Etapa 4*). |
| **Correlação de Pearson** | Mede relação **linear** entre duas variáveis contínuas, de −1 a +1. | Heatmap das diffs: gold ↔ kills = 0,92. |
| **Correlação ponto-bisserial** | Pearson entre uma variável contínua e uma binária (0/1). | Cada diff vs `blue_wins`: gold diff r = 0,51. |
| **Multicolinearidade** | Variáveis explicativas muito correlacionadas entre si. Torna coeficientes de regressão instáveis, embora não atrapalhe a previsão. | Bloco econômico: gold, kills, xp e level. |
| **VIF (Variance Inflation Factor)** | Quanto a variância de um coeficiente é inflada pela colinearidade. 1 = nenhuma; > 5 merece atenção; > 10 é alta. | `total_gold_diff` VIF = 23. |
| **Intervalo de confiança (IC)** | Faixa plausível para o valor real de uma estimativa. IC 95%: o método acerta em 95% das amostras. | Taxas de vitória por faixa, com IC. |
| **IC de Wilson** | Método de IC para proporções que funciona bem com amostras pequenas e taxas perto de 0% ou 100%. | `win_rate_by_bins` / `win_rate_by_group`. |
| **Hipótese nula (H0) / alternativa (H1)** | H0 = "não há efeito"; H1 = "há efeito". O teste mede se os dados contradizem H0. | Notebook 02 (*Etapa 3*). |
| **p-valor** | Probabilidade de ver um resultado tão extremo quanto o observado **se H0 fosse verdadeira**. Não é "a probabilidade de H0 ser verdade". | Abaixo de 0,05 → rejeita H0 (convenção). |
| **Tamanho de efeito** | **Quanto** o efeito importa, independentemente do p-valor. Com 10 mil partidas, até efeitos minúsculos ficam "significativos". | Ex.: "dragão vale X de gold" (*Etapa 3*). |
| **Teste binomial** | Testa se uma proporção observada difere de um valor esperado. | 49,9% de vitórias azuis vs 50% (*Etapa 3*). |
| **Qui-quadrado** | Testa associação entre duas variáveis categóricas. | First blood × vitória, bruto (*Etapa 3*). |
| **Confusão (confounding)** | Uma terceira variável causa as duas que parecem relacionadas. | Quem está na frente em gold pega mais dragão **e** vence mais. O dragão parece mais forte do que é. |
| **Variável de controle** | Variável incluída no modelo para "segurar" o efeito de um confundidor. | `gold_diff` nos logits do Notebook 02. |
| **Bootstrap** | Reamostrar os dados com reposição milhares de vezes para estimar a incerteza de qualquer estatística. | IC do "valor em gold" de um objetivo (*Etapa 3*) e das métricas finais (*Etapa 6*). |

## 4. Modelagem

| Termo | O que é | No projeto |
|---|---|---|
| **Feature** | Variável de entrada do modelo. | Diffs azul − vermelho + `blue_first_blood`. |
| **Alvo (target)** | O que o modelo prevê. | `blue_wins` (0/1). |
| **Feature engineering** | Criar ou transformar variáveis para o modelo. | `build_features()`, diffs e cap de wards (*Etapa 4*). |
| **Diff (diferença)** | Estatística do azul − mesma estatística do vermelho. Positivo = vantagem azul. | `total_gold_diff = blue_total_gold - red_total_gold`. |
| **Simetria de lados** | Inverter azul ↔ vermelho deve inverter o sinal de todas as diffs. | Testada em `test_features.py`. |
| **Holdout / conjunto de teste** | Parte dos dados guardada e usada **uma única vez** na avaliação final. | 20% estratificado, criado na gold (*Etapa 4*), usado só na *Etapa 6*. |
| **Estratificação** | Dividir os dados mantendo a proporção do alvo igual em cada parte. | Split e folds da CV. |
| **Validação cruzada (CV) k-fold** | Divide o treino em k partes, treina em k−1 e valida na restante, k vezes. A média é a estimativa de desempenho. | 5-fold estratificada para seleção e tuning (*Etapa 5*). |
| **Pipeline (sklearn)** | Encadeia pré-processamento e modelo num só objeto. Ajustes (média, percentil, escala) são calculados **só no treino** de cada fold. | Cap de wards + scaler + modelo. Previne vazamento. |
| **Baseline** | Modelo trivial de referência: um modelo real precisa superá-lo. | `DummyClassifier` (~50%) e "quem tem mais gold vence" (72,3%). |
| **Regressão logística** | Modelo linear que produz probabilidade via função sigmoide. Interpretável. | Candidato principal; a curva gold × vitória já é uma sigmoide. |
| **Logit** | Log da razão de chances: log(p / (1 − p)). A regressão logística é linear no logit. | Coeficientes β do Notebook 02. |
| **Regularização (L1/L2)** | Penaliza coeficientes grandes para reduzir overfitting e estabilizar sob colinearidade. | Parâmetro `C` da regressão logística. |
| **StandardScaler** | Padroniza cada feature para média 0 e desvio 1. Necessário para modelos lineares regularizados. | Dentro do Pipeline da LR. |
| **Random Forest** | Muitas árvores de decisão treinadas em amostras diferentes, com votos combinados. | Candidato (*Etapa 5*). |
| **LightGBM (gradient boosting)** | Árvores construídas em sequência, cada uma corrigindo o erro das anteriores. Forte em dados tabulares. | Candidato (*Etapa 5*). |
| **Overfitting** | O modelo decora o treino e generaliza mal. | Detectado pela diferença entre treino e CV. |
| **Hiperparâmetro** | Configuração do modelo definida antes do treino (ex.: profundidade da árvore). | Ajustados com Optuna. |
| **Optuna** | Biblioteca de busca inteligente de hiperparâmetros (busca bayesiana, *TPE*). | *Etapa 6*. |
| **MLflow** | Registro de experimentos: parâmetros, métricas e artefatos de cada treino. **Model Registry** versiona o modelo escolhido. | `mlruns/` local (*Etapa 5–6*). |
| **Artefato** | Arquivo gerado pelo treino (modelo serializado, figura, métricas). | `models/` e `reports/metrics/final.json`. |
| **joblib** | Serializa objetos Python (como o Pipeline treinado) em disco. | Exportação do modelo final. |

## 5. Avaliação

| Termo | O que é | No projeto |
|---|---|---|
| **ROC-AUC** | Probabilidade de o modelo dar nota maior a uma vitória do que a uma derrota, escolhidas ao acaso. 0,5 = sorteio; 1,0 = perfeito. Independe do limiar. | **Métrica principal.** Benchmark só com gold ≈ 0,80; teto realista ≈ 0,82. |
| **Log loss** | Penaliza probabilidades confiantes e erradas. Mede a **qualidade da probabilidade**, não só do acerto. | Importante porque o produto é uma barra de win-probability. |
| **Brier score** | Erro quadrático médio entre a probabilidade prevista e o resultado (0/1). Menor é melhor. | Mede calibração + discriminação. |
| **Acurácia** | % de previsões corretas (com limiar de 0,5). | Comparável ao baseline de 72,3%. |
| **F1** | Média harmônica de precisão e recall. | Reportada para completude; pouco relevante com classes balanceadas. |
| **Calibração** | Quando o modelo diz 70%, o time deve vencer ~70% das vezes. | Essencial para a barra de transmissão (*Etapa 6*). |
| **Curva de calibração** | Gráfico da probabilidade prevista × frequência real. Ideal = diagonal. | *Etapa 6*. |
| **CalibratedClassifierCV (sigmoid / isotônica)** | Recalibra as probabilidades de um modelo. Sigmoid (Platt): curva em S, robusta com poucos dados. Isotônica: flexível, precisa de mais dados. | Comparamos as duas (*Etapa 6*). |
| **Limiar (threshold)** | Probabilidade a partir da qual se prevê "vitória". Padrão = 0,5. | Faixas de decisão (*Etapa 7*). |

## 6. Interpretação e entrega

| Termo | O que é | No projeto |
|---|---|---|
| **SHAP** | Atribui a cada feature quanto ela empurrou a previsão para cima ou para baixo, numa partida específica. Baseado em teoria dos jogos (valores de Shapley). | *Etapa 7* e app. |
| **SHAP global / beeswarm** | Visão agregada: quais features mais importam e em que direção, em todas as partidas. | Notebook 03. |
| **SHAP waterfall** | Explicação de **uma** previsão: parte da média e soma a contribuição de cada feature. | Notebook 03 e simulador Streamlit. |
| **Faixa de decisão** | Agrupamento das probabilidades (< 15%, 15–35%, ...), com a taxa de acerto em cada faixa. | "Dá para cravar o resultado em X% das partidas aos 10 min" (*Etapa 7*). |
| **Model card** | Documento padronizado do modelo: uso pretendido, dados, métricas, limitações e riscos. | `reports/model_card.md` (*Etapa 7*). |
| **API REST** | Interface HTTP para outros sistemas consumirem o modelo. | FastAPI: `GET /health`, `POST /predict` (*Etapa 8*). |
| **FastAPI / Pydantic** | Framework web Python / validação de dados de entrada por tipos. | Rejeita entradas impossíveis (ex.: 2 dragões aos 10 min). |
| **Uvicorn** | Servidor que executa a aplicação FastAPI. | `make api`. |
| **Streamlit** | Framework para apps web de dados escritos só em Python. | Simulador "e se?" (*Etapa 9*). |
| **Docker / imagem / container** | Empacota código + dependências + sistema num ambiente isolado e reproduzível. | *Etapa 10*. |
| **Multi-stage build** | Dockerfile com uma etapa para construir e outra, enxuta, para rodar. Gera uma imagem menor. | *Etapa 10*. |
| **docker-compose** | Sobe vários containers juntos. | API + app (`make docker-up`). |
| **Smoke test** | Teste rápido que verifica se o sistema "liga" sem quebrar. | Treino numa amostra pequena no CI (*Etapa 10*). |
| **Data drift** | Mudança na distribuição dos dados em produção ao longo do tempo (ex.: patch novo muda o valor do dragão). | Seção "como eu monitoraria" no README. |

## 7. Colunas do dataset (vocabulário de jogo → coluna)

| Coluna (por lado, prefixo `blue_` / `red_`) | Significado aos 10 min |
|---|---|
| `wins` (só `blue_wins`) | 1 se o azul venceu a partida (alvo) |
| `wards_placed` / `wards_destroyed` | Wards colocadas / wards inimigas destruídas |
| `first_blood` | 1 se o time fez o primeiro abate |
| `kills` / `deaths` / `assists` | Abates / mortes / assistências do time |
| `elite_monsters` | Dragões + arautos (redundante) |
| `dragons` / `heralds` | Dragões / Arautos do Vale abatidos |
| `towers_destroyed` | Torres inimigas derrubadas |
| `total_gold` / `gold_per_min` | Gold total / gold por minuto (redundante) |
| `avg_level` | Nível médio dos 5 campeões |
| `total_experience` | XP total do time |
| `total_minions_killed` / `cs_per_min` | Tropas (lane CS) / CS por minuto (redundante) |
| `total_jungle_minions_killed` | Monstros da selva (jungle CS) |
| `gold_diff` / `experience_diff` | Diferença de gold / XP para o adversário (redundantes; recalculadas como diffs) |
