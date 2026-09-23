import base64
from collections import OrderedDict
from hashlib import sha256
from threading import Lock

import httpx
from pydantic import ValidationError

from app.config import settings
from app.errors import AgentError
from app.model_client import require_model_config
from app.schemas import AnalyzeRequest, ClassificationResponse
from app.storage import read_document

KEYWORDS = {
    "BANK_STATEMENT": ("bank", "statement", "对账单", "流水"),
    "PURCHASE_INVOICE": ("purchase", "supplier", "vendor", "采购"),
    "SALES_INVOICE": ("sales", "invoice", "销售", "发票"),
    "RECEIPT": ("receipt", "收据", "小票"),
    "PAYMENT_PLATFORM_REPORT": ("stripe", "paypal", "settlement", "结算"),
    "LOAN_STATEMENT": ("loan", "贷款"),
}
cache = OrderedDict()
lock = Lock()


def classify(body: AnalyzeRequest, key: str):
    fingerprint = sha256(body.model_dump_json().encode()).hexdigest()
    # ponytail: serialize classification per process for bounded idempotent replay;
    # Backend persists completed runs. Use provider idempotency for multi-replica scale.
    with lock:
        if key in cache:
            previous_hash, output = cache[key]
            if previous_hash != fingerprint:
                raise AgentError(409, "IDEMPOTENCY_CONFLICT", "Key was used with a different request")
            return output
        if settings.classification_provider == "MOCK" and settings.environment == "production":
            raise AgentError(503, "MOCK_NOT_ALLOWED", "Simulated classification is disabled in production")
        files, total = [], 0
        for document in body.documents:
            content = read_document(document, settings)
            total += len(content)
            if total > 100 * 1024 * 1024:
                raise AgentError(413, "BATCH_TOO_LARGE", "Select a smaller batch of documents")
            files.append(content)
        if settings.classification_provider == "MOCK":
            items = []
            for doc, content in zip(body.documents, files, strict=True):
                text = (doc.original_name + " " + content.decode("utf-8", errors="ignore")).casefold()
                invalid = any(word in text for word in ("invalid", "unrelated", "无效"))
                match = None if invalid else next((req for req in body.requirements if any(word in text for word in KEYWORDS.get(req.document_type, ()))), None)
                items.append({"document_id": str(doc.document_id), "category": "INVALID" if invalid else "REQUIREMENT" if match else "OTHER", "document_type": match.document_type if match else None, "requirement_id": str(match.id) if match else None, "confidence": 0.90 if match else 0.50})
            result = {"schema_version": "1", "run_id": str(body.run_id), "model_version": "mock-classifier-v1", "classifications": items}
        else:
            require_model_config(settings)
            if settings.classification_provider != "REMOTE":
                raise AgentError(503, "MODEL_NOT_CONFIGURED", "Classification is not enabled")
            payload = body.model_dump(mode="json")
            for document, content in zip(payload["documents"], files, strict=True):
                document.pop("storage_key")
                document["content_base64"] = base64.b64encode(content).decode()
            try:
                with httpx.Client(timeout=httpx.Timeout(settings.model_request_timeout_seconds, connect=settings.model_connect_timeout_seconds), follow_redirects=False, trust_env=False) as client:
                    with client.stream("POST", settings.model_api_url, json=payload, headers={"Authorization": f"Bearer {settings.model_api_key.get_secret_value()}", "Idempotency-Key": key}) as response:
                        if response.status_code != 200:
                            raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable")
                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 1024 * 1024:
                                raise AgentError(502, "MODEL_INVALID_RESPONSE", "Model response is too large")
                        result = httpx.Response(200, content=bytes(raw)).json()
            except httpx.TimeoutException:
                raise AgentError(504, "MODEL_TIMEOUT", "Remote model timed out") from None
            except httpx.HTTPError:
                raise AgentError(503, "MODEL_UNAVAILABLE", "Remote model is unavailable") from None
            except ValueError:
                raise AgentError(502, "MODEL_INVALID_RESPONSE", "Model response is not JSON") from None
        try:
            output = ClassificationResponse.model_validate(result)
            ids = [doc.document_id for doc in body.documents]
            requirements = {req.id for req in body.requirements}
            if output.run_id != body.run_id or [item.document_id for item in output.classifications] != ids:
                raise ValueError("Mismatched document IDs")
            if any(item.requirement_id and item.requirement_id not in requirements for item in output.classifications):
                raise ValueError("Unknown requirement")
        except (ValueError, ValidationError):
            raise AgentError(502, "MODEL_INVALID_RESPONSE", "Invalid classification response") from None
        cache[key] = (fingerprint, output)
        if len(cache) > 128:
            cache.popitem(last=False)
        return output
