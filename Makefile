.DEFAULT_GOAL := help
.PHONY: help setup data features train mlflow-ui tune evaluate api app test lint format docker-up

UV := uv run
export MLFLOW_DISABLE_AGENT_HINT := 1

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Instala dependências e hooks do pre-commit
	uv sync
	$(UV) pre-commit install

data: ## raw -> bronze (validado com Pandera)
	$(UV) python -m lol_win.data.ingest

features: ## bronze -> silver -> gold (split treino/teste fixo)
	$(UV) python -m lol_win.data.clean
	$(UV) python -m lol_win.data.split

train: ## Baselines + modelos com CV 5-fold no treino (MLflow)
	$(UV) python -m lol_win.models.train

mlflow-ui: ## Abre a interface do MLflow
	$(UV) mlflow ui --backend-store-uri sqlite:///mlflow.db

tune: ## Tuning (Optuna) de LR e LightGBM minimizando log loss na CV
	$(UV) python -m lol_win.models.tune

evaluate: ## Calibração, escolha final, avaliação no teste, registry e export
	$(UV) python -m lol_win.models.final

api: ## Sobe a API FastAPI
	$(UV) uvicorn lol_win.api.main:app --reload

app: ## Sobe o app Streamlit
	$(UV) streamlit run src/lol_win/app/streamlit_app.py

test: ## Roda os testes
	$(UV) pytest

lint: ## Lint + checagem de formatação
	$(UV) ruff check .
	$(UV) ruff format --check .

format: ## Corrige lint e formata
	$(UV) ruff check --fix .
	$(UV) ruff format .

docker-up: ## Sobe api + app via docker compose
	docker compose up --build
