# syntax=docker/dockerfile:1.7
# Imagem única para API (padrão) e app Streamlit (command sobrescrito no compose).

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.13.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------- builder: dependências + pipeline de dados + reconstrução do modelo ----------
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    MPLCONFIGDIR=/tmp/mpl
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group serve --no-install-project

COPY src ./src
COPY data/raw ./data/raw
COPY reports/metrics/final.json reports/metrics/tuning_results.json ./reports/metrics/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group serve

RUN .venv/bin/python -m lol_win.data.ingest \
    && .venv/bin/python -m lol_win.data.clean \
    && .venv/bin/python -m lol_win.data.split \
    && .venv/bin/python -m lol_win.models.export --output-dir /app/models

# ---------- runtime: só venv, código e modelo ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/models /app/models

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    LOL_WIN_MODEL_PATH=/app/models/model.joblib \
    MPLCONFIGDIR=/tmp/mpl \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER app
EXPOSE 8000 8501
CMD ["uvicorn", "lol_win.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
