"""Render each document page and send it to an independent DeepSeek-OCR-2 server."""

import base64
import io
from collections import OrderedDict
from hashlib import sha256
from threading import Lock
from time import monotonic

import httpx
import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.errors import AgentError
from app.telemetry import elapsed_ms, event
from time import perf_counter

MAX_PAGES = 20
MAX_PIXELS = 16_000_000
MAX_OCR_CHARS = 120_000
MAX_RESPONSE_BYTES = 512 * 1024
DEFAULT_API_URL = "https://api.novita.ai/openai/v1/chat/completions"
DEFAULT_HEALTH_URL = "https://api.novita.ai/openai/v1/models"
# ponytail: per-process OCR reuse is enough for one Agent replica; use shared encrypted storage if multi-replica cost warrants it.
_cache = OrderedDict()
_cache_lock = Lock()


def _key(settings: Settings) -> str:
    return settings.ocr_api_key.get_secret_value() or settings.novita_api_key.get_secret_value()


def _pages(content: bytes, content_type: str):
    if content_type == "application/pdf":
        try:
            pdf = pdfium.PdfDocument(content)
            if not 1 <= len(pdf) <= MAX_PAGES:
                raise AgentError(422, "DOCUMENT_PAGE_LIMIT", "PDF page count is not supported")
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    if page.get_width() * page.get_height() * 1.5**2 > MAX_PIXELS:
                        raise AgentError(422, "DOCUMENT_IMAGE_LIMIT", "Document page is too large")
                    bitmap = page.render(scale=1.5)
                    try:
                        yield bitmap.to_pil().convert("RGB")
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        except (pdfium.PdfiumError, ValueError):
            raise AgentError(422, "DOCUMENT_UNREADABLE", "PDF could not be rendered") from None
        finally:
            if "pdf" in locals():
                pdf.close()
    else:
        try:
            with Image.open(io.BytesIO(content)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise AgentError(422, "DOCUMENT_IMAGE_LIMIT", "Image is too large")
                yield source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            raise AgentError(422, "DOCUMENT_UNREADABLE", "Image could not be opened") from None


def _read_json(response: httpx.Response) -> dict:
    if response.status_code != 200:
        raise AgentError(503, "OCR_UNAVAILABLE", "OCR service is unavailable")
    raw = bytearray()
    for chunk in response.iter_bytes():
        raw.extend(chunk)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AgentError(502, "OCR_INVALID_RESPONSE", "OCR response is too large")
    try:
        payload = httpx.Response(200, content=bytes(raw)).json()
    except ValueError:
        raise AgentError(502, "OCR_INVALID_RESPONSE", "OCR response is not JSON") from None
    if not isinstance(payload, dict):
        raise AgentError(502, "OCR_INVALID_RESPONSE", "OCR response is invalid")
    return payload


def require_ocr(settings: Settings):
    if not _key(settings):
        raise AgentError(503, "OCR_NOT_CONFIGURED", "Novita OCR is not configured")


def check_ocr_ready(settings: Settings):
    require_ocr(settings)
    url = settings.ocr_health_url or (settings.ocr_api_url.removesuffix("/chat/completions") + "/models" if settings.ocr_api_url else DEFAULT_HEALTH_URL)
    headers = {"Authorization": f"Bearer {_key(settings)}"}
    try:
        with httpx.Client(timeout=settings.model_connect_timeout_seconds, follow_redirects=False, trust_env=False) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    raise AgentError(503, "OCR_UNAVAILABLE", "OCR service is unavailable")
    except httpx.TimeoutException:
        raise AgentError(504, "OCR_TIMEOUT", "OCR service timed out") from None
    except httpx.HTTPError:
        raise AgentError(503, "OCR_UNAVAILABLE", "OCR service is unavailable") from None


def ocr_document(content: bytes, content_type: str, settings: Settings, deadline: float | None = None,
                 run_id: str | None = None, document_id: str | None = None) -> str:
    require_ocr(settings)
    cache_key = (sha256(content).digest(), content_type, settings.ocr_model, settings.ocr_api_url)
    with _cache_lock:
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            event("ocr_cache_hit", run_id=run_id, document_id=document_id, text_chars=len(_cache[cache_key]))
            return _cache[cache_key]
    headers = {"Authorization": f"Bearer {_key(settings)}"}
    deadline = deadline or monotonic() + settings.model_request_timeout_seconds - 10
    output = []
    current_page = 0
    page_started = perf_counter()
    provider_status = None
    try:
        with httpx.Client(follow_redirects=False, trust_env=False) as client:
            for index, image in enumerate(_pages(content, content_type), start=1):
                current_page, page_started, provider_status = index, perf_counter(), None
                event("ocr_page_started", run_id=run_id, document_id=document_id, page=index,
                      model=settings.ocr_model)
                if monotonic() >= deadline:
                    raise AgentError(504, "OCR_TIMEOUT", "OCR processing timed out")
                image.thumbnail((1800, 1800))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                payload = {"model": settings.ocr_model, "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
                    {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                ]}], "max_tokens": 4096, "temperature": 0, "top_k": 0, "stream": False}
                timeout = httpx.Timeout(min(60, max(1, deadline - monotonic())), connect=settings.model_connect_timeout_seconds)
                with client.stream("POST", settings.ocr_api_url or DEFAULT_API_URL, json=payload, headers=headers, timeout=timeout) as response:
                    provider_status = response.status_code
                    result = _read_json(response)
                try:
                    choice = result["choices"][0]
                    text = choice["message"]["content"]
                    if choice.get("finish_reason") not in ("stop", None) or not isinstance(text, str) or not text.strip():
                        raise ValueError
                except (KeyError, IndexError, TypeError, ValueError):
                    raise AgentError(502, "OCR_INVALID_RESPONSE", "OCR output is incomplete") from None
                output.append(f"[page {index}]\n{text.strip()}")
                if sum(map(len, output)) > MAX_OCR_CHARS:
                    raise AgentError(413, "OCR_TEXT_LIMIT", "OCR text exceeds the analysis limit")
                event("ocr_page_finished", run_id=run_id, document_id=document_id, page=index,
                      status="ok", provider_status=provider_status, text_chars=len(text),
                      duration_ms=elapsed_ms(page_started))
                current_page = 0
    except AgentError as exc:
        if current_page:
            event("ocr_page_finished", run_id=run_id, document_id=document_id, page=current_page,
                  status="failed", provider_status=provider_status, error_code=exc.code,
                  duration_ms=elapsed_ms(page_started))
        raise
    except httpx.TimeoutException:
        if current_page:
            event("ocr_page_finished", run_id=run_id, document_id=document_id, page=current_page,
                  status="failed", error_code="OCR_TIMEOUT", duration_ms=elapsed_ms(page_started))
        raise AgentError(504, "OCR_TIMEOUT", "OCR service timed out") from None
    except httpx.HTTPError:
        if current_page:
            event("ocr_page_finished", run_id=run_id, document_id=document_id, page=current_page,
                  status="failed", error_code="OCR_UNAVAILABLE", duration_ms=elapsed_ms(page_started))
        raise AgentError(503, "OCR_UNAVAILABLE", "OCR service is unavailable") from None
    text = "\n\n".join(output)
    with _cache_lock:
        _cache[cache_key] = text
        _cache.move_to_end(cache_key)
        if len(_cache) > 128:
            _cache.popitem(last=False)
    return text
