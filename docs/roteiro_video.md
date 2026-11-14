# Roteiro do vídeo: LoL Win Predictor (5–7 min)

**Formato:** gravação de tela com narração em off (sem câmera). O formato final será decidido depois; o roteiro serve para qualquer edição.
**Tom:** de jogador para jogador, com rigor de cientista de dados. Frases curtas. Números arredondados na fala; os exatos ficam na tela.
**Preparação:**
- `docker compose up` rodando, com o app em `localhost:8501` e a API em `localhost:8000/docs`;
- notebooks abertos já executados;
- README aberto no GitHub.

---

## Cena 1 · Gancho (0:00–0:30)

**Visual:** waterfall SHAP da maior virada (`reports/figures/interp_shap_waterfall_virada.png`), com zoom no título "P(azul) = 97% · vermelho venceu".

**Fala:**
> "Dez minutos de jogo. Time azul com seis mil e setecentos de gold de vantagem. Meu modelo dá 97% de chance de vitória para ele... e o vermelho vira. Isso é erro do modelo? Ou é o jogo? Esse projeto responde essa pergunta e mais algumas que todo jogador de LoL já se fez."

---

## Cena 2 · O problema (0:30–1:10)

**Visual:** topo do README (pergunta de negócio e os 3 usos) e um corte rápido para uma linha do CSV.

**Fala:**
> "O cenário: eu sou o cientista de dados que responde ao Tech Lead de dados da Riot. A pergunta é: aos 10 minutos, qual a probabilidade do lado azul vencer? Isso serve para três coisas: a barra de win-probability nas transmissões, um indicador de 'jogo decidido' perto do surrender aos 15, e entender quanto vale cada objetivo.
> Os dados são quase dez mil partidas ranqueadas de elo alto, cada uma com uma foto do jogo aos 10 minutos: gold, XP, kills, dragões, torres, wards. Não tem composição nem nada depois dos 10 minutos, e isso vai importar no final."

---

## Cena 3 · Tudo vira gold (1:10–2:00)

**Visual:** `eda_win_rate_by_gold_diff.png`. Depois `hyp_raw_vs_adjusted_odds.png`, destacando o first blood indo de 2,3 para 1,0.

**Fala:**
> "Primeira descoberta: o gold diff sozinho conta quase a história inteira. Olha essa curva: uma sigmoide quase perfeita. Mil de gold dá 57%, três mil dá 83%, cinco mil dá 98%.
> E o first blood? Na conta bruta, quem faz first blood tem mais que o dobro das chances de vencer. Mas quando eu comparo times com o mesmo gold, o efeito some. O first blood não decide nada: quem faz é quem já está jogando melhor, e o valor dele já está no gold. O mesmo vale para wards. Em estatística, isso tem nome: confusão."

---

## Cena 4 · Dragão e o paradoxo da torre (2:00–3:00)

**Visual:** `hyp_objective_gold_value.png` (dragão +560, arauto ~0, torre −620), depois `hyp_tower_paradox.png`.

**Fala:**
> "Então eu perguntei: quanto vale cada objetivo em gold, além do gold que ele já deu? O dragão vale uns quinhentos e sessenta. Faz sentido: ele não dá gold direto, então o placar subestima. O arauto não vale nada além do gold que gera.
> E a torre vale... negativo. Com o mesmo gold diff, quem tem torre vence menos. Não é que derrubar torre seja ruim: é que esse gold veio sem XP e sem kill. Na mesma faixa de gold, o time da torre tem quase quatro vezes menos vantagem de XP. Moral: mil de gold vindo de torre valem menos que mil vindo de luta e farm."

---

## Cena 5 · O modelo (3:00–3:50)

**Visual:** `model_cv_comparison.png`, depois a tabela de teste do README e `final_calibration_curve.png`.

**Fala:**
> "Para o modelo, comparei três famílias: regressão logística, Random Forest e LightGBM, com validação cruzada e tuning no Optuna. Venceu o mais simples: a regressão logística. Ela foi melhor em todas as métricas e já sai calibrada.
> No teste, que eu só abri uma vez, no final, a AUC foi 0,81, bem no teto realista para dados aos 10 minutos. E aqui vai uma leitura honesta: em acurácia, ele empata com a regra 'quem tem mais gold vence'. A diferença é que a regra só sabe dizer 'sim' ou 'não'. O modelo diz quanto: e quando ele diz 70%, o time ganha mais ou menos 70% das vezes."

---

## Cena 6 · Demo ao vivo (3:50–5:10)

**Visual:** app Streamlit. Sequência:
1. "partida parelha" → barra em ~50%;
2. "+1.000 gold azul" → mostrar o Δ;
3. "+1 dragão azul" → Δ positivo;
4. "+1 torre azul" → Δ ≈ −5 p.p. (paradoxo). Sequência conferida: 50% → +12 → +7 → −5 p.p.;
5. waterfall do cenário;
6. "partida real aleatória";
7. corte rápido para o `/docs` da API, executando um `POST /predict`;
8. tentar um segundo dragão → mensagem de erro.

**Fala:**
> "Isso tudo vira produto. Aqui é o simulador: começo com uma partida parelha, cinquenta a cinquenta. Mil de gold para o azul... sobe. Um dragão... sobe mais. E uma torre, sem o gold dela... cai uns cinco pontos: o paradoxo aparecendo ao vivo.
> Embaixo, o modelo explica cada previsão: quanto o gold empurrou, quanto o XP, quanto o dragão.
> E a mesma coisa está numa API, pronta para uma transmissão consumir. Ela valida as regras do jogo: se eu tentar dois dragões aos 10 minutos, ela recusa."

---

## Cena 7 · Decisão e as viradas (5:10–6:10)

**Visual:** `interp_decision_bands.png`, depois `interp_comebacks_profile.png`.

**Fala:**
> "E o que a Riot faz com isso? Aos 10 minutos, dá para cravar o vencedor em mais ou menos um quarto das partidas, com 93% de acerto. Em um terço delas, o jogo está aberto de verdade: o favorito acerta pouco mais que uma moeda.
> Voltando àquela virada do começo: entre os jogos em que o modelo deu mais de 80%, cerca de 12% viraram. É exatamente o que um modelo calibrado prevê. E quando eu comparo as viradas com as vantagens que se mantiveram, na mesma faixa de confiança, não existe diferença em nada que o dado enxerga. A virada não é um ponto cego do modelo: é o jogo acontecendo depois dos 10 minutos. Por isso o indicador deve dizer 'muito provável', e nunca 'acabou'."

---

## Cena 8 · Bastidores e fechamento (6:10–6:50)

**Visual:** diagrama Mermaid do README, os jobs verdes do CI no GitHub Actions, `docker compose up` no terminal e o link do repositório.

**Fala:**
> "Por trás disso: contratos de dados que barram partida impossível, o mesmo pipeline do treino até a API, mais de 130 testes, CI e Docker. Sobe tudo com um comando.
> O próximo passo é sair do snapshot: dados minuto a minuto da Riot API e composição dos times. É aí que estão as viradas. O código está no GitHub; o link está na descrição. Obrigado!"

---

### Checklist de gravação
- [ ] Resolução 1920×1080, zoom do navegador em ~125% para os números ficarem legíveis
- [ ] Testar a sequência da demo antes. Os Δ dependem da ordem dos cliques: começar sempre pela "partida parelha"
- [ ] Fechar outras abas e notificações
- [ ] Narração gravada separada, cena por cena (mais fácil de editar)
- [ ] Cortar tempos de loading na edição
