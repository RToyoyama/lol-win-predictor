# Model Card: LoL Win Predictor (10 minutos)

| | |
|---|---|
| **Modelo** | Regressão logística (elastic net, `C = 0,24`, `l1_ratio = 0,98`, solver `saga`), sem calibração extra |
| **Versão** | MLflow Model Registry `lol-win-predictor` v1, alias `champion` · artefato `models/model.joblib` |
| **Entrada** | 25 estatísticas brutas dos dois times aos 10 min (`config.INPUT_COLUMNS`) |
| **Saída** | `P(lado azul vence)` ∈ [0, 1] |
| **Autor / contexto** | Projeto final EBAC (Cientista de Dados). Stakeholder fictício: Tech Lead de dados da Riot Games |

## 1. Uso pretendido

**Pergunta:** *aos 10 minutos de partida, qual a probabilidade de o lado azul vencer?*

| Uso | Como usar a saída |
|---|---|
| Barra de win-probability em transmissões | Exibir a probabilidade (arredondada a 5 p.p.) |
| Indicador de "jogo decidido" (contexto do surrender aos 15) | Sinalizar quando o favorito tem ≥ 85% ("muito provável", nunca "acabou") |
| Análise de equilíbrio de lados e valor de objetivos | Ler junto com o Notebook 02 (valores em gold com IC) |

**Fora de escopo:**
- Previsão antes dos 10 minutos ou depois deles.
- Previsão de pré-jogo (composição, picks/bans).
- Partidas fora de ranked de elo alto.
- Decisões automáticas com consequência para jogadores, como punição ou matchmaking.
- Apostas.

## 2. Dados

- **Fonte:** `Base_M43_Pratique_LOL_RANKED_WIN.csv`, com 9.879 partidas ranqueadas de elo alto e um snapshot aos 10 minutos por partida. Patch e região não são informados.
- **Alvo:** `blue_wins`, balanceado (azul vence 49,9%).
- **Pipeline:** raw → bronze (contrato Pandera, 21 regras de consistência) → silver (13 colunas redundantes removidas) → gold (split 80/20 estratificado, fixo, `random_state = 42`).
- **Treino / teste:** 7.903 / 1.976 partidas. O teste foi usado **uma única vez**, na avaliação final.

## 3. Features (dentro do `Pipeline`)

O modelo recebe as stats brutas e calcula internamente as **diferenças azul − vermelho** de:
- gold;
- XP;
- nível médio;
- kills;
- assists;
- dragões;
- arautos;
- torres;
- CS de lane;
- CS de jungle.

Ficam de fora o conjunto `diffs_core`, as wards e o first blood: o Notebook 02 mostrou que eles **não têm efeito além do gold**, e a CV confirmou. O artefato exportado contém toda a transformação, então treino, API e app usam o mesmo código (sem *training-serving skew*).

## 4. Desempenho

### Teste (holdout, IC 95% por bootstrap, B = 1.000)

| Métrica | Modelo | Regra "quem tem mais gold vence" |
|---|---|---|
| **ROC-AUC** | **0,807** [0,789; 0,825] | 0,718 |
| Log loss | 0,531 [0,510; 0,551] | 10,16 |
| Brier | 0,180 [0,171; 0,188] | 0,282 |
| Acurácia (limiar 0,5) | 71,7% [69,9; 73,6] | 71,8% |
| ECE (calibração) | 0,027 | — |

Na CV 5-fold do treino, o modelo teve AUC 0,812, log loss 0,528 e acurácia 73,7%. Os números do teste são consistentes com a CV. O benchmark de uma LR só com gold diff foi AUC 0,803.

**Leitura honesta:** em **acurácia**, o modelo empata com a regra do gold. O ganho está na **probabilidade**: ele diz *quão* decidida está a partida, com calibração boa (ECE 0,03), enquanto a regra só dá 0% ou 100%.

### Faixas de decisão (teste)

| P(azul) | % das partidas | Taxa real de vitória azul | Acerto do favorito |
|---|---|---|---|
| < 15% | 12,8% | 5,9% | 94,1% [90,4; 96,4] |
| 15–35% | 21,2% | 27,7% | 72,3% [67,8; 76,4] |
| 35–65% | 31,3% | 50,3% | 53,7% [49,8; 57,6] |
| 65–85% | 21,0% | 71,3% | 71,3% [66,7; 75,4] |
| > 85% | 13,8% | 91,5% | 91,5% [87,6; 94,3] |

> **Aos 10 minutos, dá para cravar o vencedor em ~27% das partidas, com ~93% de acerto.
> Em ~31% delas o jogo está em aberto: o favorito acerta só ~54%.**

As predições out-of-fold do treino reproduzem as mesmas proporções, então as faixas são estáveis.

## 5. Interpretação (SHAP, teste)

| Feature | Participação na média de \|SHAP\| |
|---|---|
| Gold | 54% |
| XP | 18% |
| Dragões | 14% |
| Demais (CS, assists, kills, nível, torres, arautos) | 14% |

- O **dragão** é o único objetivo com peso próprio relevante. Isso bate com a estimativa de **~560 de gold** de valor além do gold já gerado (Notebook 02, IC 464–657).
- **Viradas:** entre as partidas com favorito ≥ 80%, **11,6% viraram**, praticamente os 11,2% esperados para um modelo calibrado. Dentro da mesma faixa de confiança, viradas e vantagens mantidas **não se diferenciam** em nenhuma feature disponível. As viradas refletem o que acontece **depois** dos 10 minutos e não estão nos dados.

## 6. Limitações

- **Snapshot único:** não há composição, picks/bans, MMR nem estado minuto a minuto. O teto prático com esses dados é de AUC ≈ 0,81–0,82.
- **População:** só elo alto, com patch e região desconhecidos. Não dá para garantir a transferência para outros elos ou patches.
- **Colinearidade:** gold, XP, kills e nível são fortemente correlacionados (VIF 6–23). Coeficientes e valores SHAP individuais desse bloco **não devem ser lidos isoladamente**; só o bloco como um todo é robusto.
- **Associação, não causa:** os valores de objetivos (Notebook 02) são associações controladas pelo gold, não efeitos causais.
- **Acurácia:** limitada a cerca de 72% no limiar de 0,5, porque cerca de 31% das partidas estão genuinamente em aberto aos 10 minutos.

## 7. Considerações éticas e de uso

- **Viés de exibição:** uma barra de probabilidade pode desmotivar jogadores e espectadores ("acabou"). Como mitigação, a comunicação deve usar "muito provável" e lembrar que **cerca de 1 em cada 14** jogos com favorito acima de 85% vira.
- **Surrender:** o indicador é **contexto**, não recomendação. A decisão continua sendo do time.
- **Dados pessoais:** não há dados de jogadores. O `game_id` não é usado como feature.
- **Apostas:** o modelo não foi validado para esse uso e não deve ser usado para isso.

## 8. Monitoramento recomendado (produção)

| O que monitorar | Como | Gatilho |
|---|---|---|
| Drift de entrada | PSI das diffs de gold, XP e dragões por patch | PSI > 0,2 |
| Calibração | ECE e curva de calibração semanais com resultados reais | ECE > 0,05 |
| Discriminação | AUC móvel (janela de 1–2 semanas) | Queda > 0,02 vs. 0,807 |
| Mudança de patch | Reestimar o valor do dragão e retreinar | Patch com mudança em objetivos ou gold |

## 9. Reprodutibilidade

```bash
make setup && make data features train tune evaluate
```

- Seeds fixas (`RANDOM_STATE = 42`) e dependências travadas no `uv.lock`.
- Experimentos no MLflow (`make mlflow-ui`).
- Métricas em `reports/metrics/` e figuras em `reports/figures/`.
