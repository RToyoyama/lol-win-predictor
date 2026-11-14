.DEFAULT_GOAL := help
.PHONY: help setup data features train tune evaluate api app test lint format docker-up

UV := uv run
export MLFLOW_DISABLE_AGENT_HINT := 1

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Instala dependências e hooks do pre-commit
	uv sync
	$(UV) pre-commit install

data: ## raw -> bronze (validado com Pandera)
	@echo "TODO: Etapa 1"

features: ## bronze -> silver -> gold + split treino/teste
	@echo "TODO: Etapa 4"

train: ## Treina baselines e modelos com CV (MLflow)
	@echo "TODO: Etapa 5"

tune: ## Tuning com Optuna
	@echo "TODO: Etapa 6"

evaluate: ## Calibração e avaliação final no teste
	@echo "TODO: Etapa 6"

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
