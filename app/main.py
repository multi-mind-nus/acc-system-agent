import json
import logging
import re
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.config import settings
from app.errors import AgentError
from app.model_client import check_model_ready
from app.schemas import AnalyzeRequest
from app.classification import classify
from app.analysis_schemas import ReviewRequest
from app.review import review

logger = logging.getLogger("agent.requests")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
# HTTP libraries otherwise log remote URLs; remote responses may contain secrets.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="acc-system-agent", version=settings.app_version)


def error(request: Request, status: int, code: str, message: str):
    return JSONResponse(status_code=status, content={
        "code": code, "message": message, "details": None,
        "request_id": request.state.request_id,
    })


@app.exception_handler(AgentError)
def agent_error(request: Request, exc: AgentError):
    return error(request, exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
def invalid_request(request: Request, exc: RequestValidationError):
    return error(request, 422, "VALIDATION_ERROR", "Request validation failed")


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return error(request, exc.status_code, "HTTP_ERROR", "Request could not be handled")


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request.state.request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9-]{1,64}", supplied_id) else str(uuid4())
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Do not log arbitrary exception strings: SDK errors may include secrets.
        response = error(request, 500, "INTERNAL_ERROR", "An unexpected error occurred")
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request.state.request_id,
        "method": request.method,
        "status_code": response.status_code,
        "duration_ms": round((perf_counter() - started) * 1000, 2),
        "run_id": getattr(request.state, "run_id", None),
    }))
    return response


@app.get("/health/live")
def live():
    return {"status": "ok", "version": settings.app_version}


@app.get("/health/ready")
def ready():
    if not settings.document_path.is_dir():
        raise AgentError(503, "DOCUMENT_STORAGE_UNAVAILABLE", "Document storage is unavailable")
    if settings.classification_provider == "MOCK" and settings.environment != "production":
        return {"status": "ok", "provider": "MOCK"}
    check_model_ready(settings)
    return {"status": "ok", "dependencies": {"documents": "ok", "model": "ok"}}


@app.post("/v1/analyze")
def analyze(body: AnalyzeRequest | ReviewRequest, request: Request, idempotency_key: str = Header(min_length=1, max_length=128)):
    request.state.run_id = str(body.run_id)
    if not re.fullmatch(re.escape(str(body.run_id)) + r":[0-9]{1,3}", idempotency_key):
        raise AgentError(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency key must match run_id:turn")
    if body.purpose == "CLASSIFY":
        return classify(body, idempotency_key)
    if isinstance(body, ReviewRequest):
        return review(body, idempotency_key)
