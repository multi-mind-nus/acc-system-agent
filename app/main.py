import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.config import settings
from app.errors import AgentError
from app.model_client import check_model_ready
from app.deepseek import check_deepseek_ready
from app.ocr import check_ocr_ready
from app.schemas import AnalyzeRequest
from app.classification import classify
from app.analysis_schemas import ReviewRequest
from app.review import review
from app.inline import InlineClassifyRequest, InlineReviewRequest, decode_inline
from app.telemetry import elapsed_ms, event, logger

# HTTP libraries otherwise log remote URLs; remote responses may contain secrets.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(title="acc-system-agent", version=settings.app_version)


def error(request: Request, status: int, code: str, message: str):
    request.state.error_code = code
    if getattr(request.state, "run_id", None):
        event("analysis_failed", run_id=request.state.run_id,
              purpose=getattr(request.state, "purpose", None), error_code=code)
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
    if request.url.path not in ("/health/live", "/health/ready") or response.status_code >= 400:
        event("request_finished", request_id=request.state.request_id, method=request.method,
              status_code=response.status_code, duration_ms=elapsed_ms(started),
              run_id=getattr(request.state, "run_id", None), purpose=getattr(request.state, "purpose", None),
              error_code=getattr(request.state, "error_code", None))
    return response


@app.get("/health/live")
def live():
    return {"status": "ok", "version": settings.app_version}


@app.get("/health/ready")
def ready():
    if not settings.document_path.is_dir():
        raise AgentError(503, "DOCUMENT_STORAGE_UNAVAILABLE", "Document storage is unavailable")
    providers = (settings.classification_provider, settings.review_provider)
    if "MOCK" in providers and settings.environment == "production":
        raise AgentError(503, "MOCK_NOT_ALLOWED", "Simulated analysis is disabled in production")
    if "MOCK" in providers and all(provider in ("MOCK", "DISABLED") for provider in providers):
        return {"status": "ok", "provider": "MOCK"}
    if "DEEPSEEK" in providers:
        check_ocr_ready(settings)
        check_deepseek_ready(settings)
        return {"status": "ok", "dependencies": {"documents": "ok", "ocr": "ok", "model": "ok"}}
    check_model_ready(settings)
    return {"status": "ok", "dependencies": {"documents": "ok", "model": "ok"}}


@app.post("/v1/analyze")
def analyze(body: AnalyzeRequest | ReviewRequest, request: Request, idempotency_key: str = Header(min_length=1, max_length=128)):
    request.state.run_id = str(body.run_id)
    request.state.purpose = body.purpose
    if not re.fullmatch(re.escape(str(body.run_id)) + r":[0-9]{1,3}", idempotency_key):
        raise AgentError(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency key must match run_id:turn")
    if body.purpose == "CLASSIFY":
        return classify(body, idempotency_key)
    if isinstance(body, ReviewRequest):
        return review(body, idempotency_key)


@app.post("/v1/analyze-inline")
def analyze_inline(body: InlineClassifyRequest | InlineReviewRequest, request: Request,
                   idempotency_key: str = Header(min_length=1, max_length=128), authorization: str = Header(default="")):
    """Stand-alone API for trusted callers; no shared volume or business database required."""
    secret = settings.agent_api_key.get_secret_value()
    if not secret:
        raise AgentError(503, "AGENT_AUTH_NOT_CONFIGURED", "Standalone API is not configured")
    from secrets import compare_digest
    if not compare_digest(authorization, f"Bearer {secret}"):
        raise AgentError(401, "UNAUTHORIZED", "Invalid API credentials")
    request.state.run_id = str(body.run_id)
    request.state.purpose = body.purpose
    expected = f"{body.run_id}:{body.turn if body.purpose == 'REVIEW' else 0}"
    if idempotency_key != expected:
        raise AgentError(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency key must match run_id:turn")
    parsed, files = decode_inline(body)
    return classify(parsed, idempotency_key, files) if body.purpose == "CLASSIFY" else review(parsed, idempotency_key, files)
