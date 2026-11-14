"""API FastAPI: probabilidade de vitória aos 10 minutos a partir das stats brutas."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from sklearn.base import BaseEstimator

from lol_win import __version__
from lol_win.api.schemas import (
    Band,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    MatchStats,
    PredictionResponse,
    to_input_frame,
)
from lol_win.config import MODEL_PATH
from lol_win.models.predict import load_metadata, load_model, predict_proba, probability_band

logger = logging.getLogger(__name__)


def _model_version(metadata: dict[str, Any]) -> str:
    registry = metadata.get("mlflow", {})
    if "registered_model" in registry and "version" in registry:
        return f"{registry['registered_model']}:v{registry['version']}"
    return "unknown"


def _to_response(p_blue: float, model_version: str) -> PredictionResponse:
    label, description = probability_band(p_blue)
    return PredictionResponse(
        p_blue_win=round(p_blue, 4),
        p_red_win=round(1 - p_blue, 4),
        favorite="blue" if p_blue >= 0.5 else "red",
        faixa=Band(label=label, description=description),
        model_version=model_version,
    )


def get_model(request: Request) -> BaseEstimator:
    """Dependência: devolve o modelo carregado no startup ou responde 503.

    Args:
        request: Requisição atual.

    Returns:
        Estimador ajustado.

    Raises:
        HTTPException: 503 se o modelo não foi carregado.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo não carregado")
    return model


ModelDep = Annotated[BaseEstimator, Depends(get_model)]


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    """Cria a aplicação, carregando o modelo uma única vez no startup.

    Args:
        model_path: Arquivo ``.joblib`` do modelo (``metadata.json`` ao lado, se existir).

    Returns:
        Aplicação FastAPI.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.metadata = None, {}
        try:
            app.state.model = load_model(model_path)
        except FileNotFoundError:
            logger.error("Modelo indisponível em %s; API sobe em modo 503", model_path)
        metadata_path = model_path.with_name("metadata.json")
        if app.state.model is not None and metadata_path.exists():
            app.state.metadata = load_metadata(metadata_path)
        app.state.model_version = _model_version(app.state.metadata)
        yield

    app = FastAPI(
        title="LoL Win Predictor API",
        version=__version__,
        description=(
            "Probabilidade de vitória do lado azul aos 10 minutos de uma partida ranqueada, "
            "a partir das estatísticas brutas dos dois times."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, response: Response) -> HealthResponse:
        """Readiness: 200 com o modelo carregado, 503 caso contrário."""
        loaded = request.app.state.model is not None
        if not loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            model_version=request.app.state.model_version if loaded else None,
            variant=request.app.state.metadata.get("variant"),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(match: MatchStats, model: ModelDep, request: Request) -> PredictionResponse:
        """Probabilidade de vitória de uma partida."""
        p_blue = float(predict_proba(to_input_frame([match]), model)[0])
        return _to_response(p_blue, request.app.state.model_version)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, model: ModelDep, request: Request) -> BatchResponse:
        """Probabilidades de um lote de partidas, na mesma ordem."""
        probabilities = predict_proba(to_input_frame(batch.matches), model)
        version = request.app.state.model_version
        return BatchResponse(predictions=[_to_response(float(p), version) for p in probabilities])

    return app


app = create_app()
