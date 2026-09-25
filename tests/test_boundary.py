import logging
import os
from hashlib import sha256
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import main, model_client
from app.config import settings
from app.errors import AgentError
from app.schemas import DocumentReference
from app.storage import read_document


@pytest.fixture
def document(tmp_path, monkeypatch):
    root = tmp_path / "documents"
    root.mkdir()
    content = b"%PDF-1.7\nprivate document body"
    (root / "sample").write_bytes(content)
    monkeypatch.setattr(settings, "document_path", root)
    monkeypatch.setattr(settings, "model_api_url", "")
    monkeypatch.setattr(settings, "model_health_url", "")
    monkeypatch.setattr(settings, "model_api_key", SecretStr(""))
    return DocumentReference(document_id=uuid4(), storage_key="sample", content_type="application/pdf", sha256=sha256(content).hexdigest())


def test_regular_file_and_trust_boundary(document, tmp_path):
    assert read_document(document, settings).startswith(b"%PDF-")
    outside = tmp_path / "private"
    outside.write_bytes(b"private")
    (settings.document_path / "link").symlink_to(outside)
    (settings.document_path / "dir").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(settings.document_path / "fifo")
    for key in ("../private", str(outside), "dir/private", "link", "fifo", "missing", "a/../sample", "a\\sample", "sample/", "\0"):
        with pytest.raises(AgentError):
            read_document(document.model_copy(update={"storage_key": key}), settings)


@pytest.mark.parametrize("changes,code", [
    ({"sha256": "0" * 64}, "DOCUMENT_HASH_MISMATCH"),
    ({"content_type": "image/png"}, "DOCUMENT_TYPE_MISMATCH"),
])
def test_hash_and_mime(document, changes, code):
    with pytest.raises(AgentError) as caught:
        read_document(document.model_copy(update=changes), settings)
    assert caught.value.code == code


def test_size_limit(document, monkeypatch):
    monkeypatch.setattr(settings, "max_document_bytes", 4)
    with pytest.raises(AgentError) as caught:
        read_document(document, settings)
    assert caught.value.code == "DOCUMENT_SIZE_INVALID"


def test_health_and_analysis_are_explicitly_unconfigured(document):
    with TestClient(main.app) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_NOT_CONFIGURED"
        run_id = str(uuid4())
        body = {"run_id": run_id, "purpose": "CLASSIFY", "documents": [document.model_dump(mode="json")]}
        assert client.post("/v1/analyze", json=body).status_code == 422
        assert client.post("/v1/analyze", json=body, headers={"Idempotency-Key": "wrong:0"}).json()["code"] == "INVALID_IDEMPOTENCY_KEY"
        response = client.post("/v1/analyze", json=body, headers={"Idempotency-Key": f"{run_id}:0"})
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_NOT_CONFIGURED"
        assert set(response.json()) == {"code", "message", "details", "request_id"}


@pytest.mark.parametrize("result,expected", [
    (httpx.Response(200, json={"status": "ok"}), (200, None)),
    (httpx.Response(200, json={"status": "loading"}), (503, "MODEL_NOT_READY")),
    (httpx.Response(200, content=b"not JSON"), (502, "MODEL_INVALID_RESPONSE")),
    (httpx.Response(200, content=b"x" * 65537), (502, "MODEL_INVALID_RESPONSE")),
    (httpx.Response(401), (503, "MODEL_UNAVAILABLE")),
    (httpx.Response(302, headers={"Location": "http://other.test"}), (503, "MODEL_UNAVAILABLE")),
    (httpx.ReadTimeout("sensitive remote detail"), (504, "MODEL_TIMEOUT")),
    (httpx.ConnectError("sensitive remote detail"), (503, "MODEL_UNAVAILABLE")),
])
def test_remote_failures_are_sanitized_and_not_retried(document, monkeypatch, caplog, result, expected):
    monkeypatch.setattr(settings, "model_api_url", "https://model.test/analyze")
    monkeypatch.setattr(settings, "model_health_url", "https://model.test/health")
    monkeypatch.setattr(settings, "model_api_key", SecretStr("sensitive-key"))
    monkeypatch.setattr(main.logger, "propagate", True)
    calls = []

    def respond(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer sensitive-key"
        if isinstance(result, Exception):
            raise result
        return result

    real_client = httpx.Client
    monkeypatch.setattr(model_client.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    with TestClient(main.app) as client, caplog.at_level(logging.INFO):
        response = client.get("/health/ready")
    assert response.status_code == expected[0]
    assert response.json().get("code") == expected[1]
    assert len(calls) == 1
    assert "sensitive" not in response.text + caplog.text
    assert "private document body" not in caplog.text


def test_configured_analysis_is_not_fake_success(document, monkeypatch):
    monkeypatch.setattr(settings, "model_api_url", "https://model.test/analyze")
    monkeypatch.setattr(settings, "model_api_key", SecretStr("test"))
    run_id = str(uuid4())
    body = {"run_id": run_id, "purpose": "REVIEW", "documents": [document.model_dump(mode="json")]}
    with TestClient(main.app) as client:
        response = client.post("/v1/analyze", json=body, headers={"Idempotency-Key": f"{run_id}:0"})
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
        assert "findings" not in response.json()
        body["documents"] *= 2
        assert client.post("/v1/analyze", json=body, headers={"Idempotency-Key": f"{run_id}:0"}).status_code == 422


def test_unexpected_error_never_leaks_secrets(document, monkeypatch, caplog):
    def broken():
        raise RuntimeError("sensitive-key private document body")

    monkeypatch.setattr(main, "check_model_ready", lambda _: broken())
    with TestClient(main.app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 500
        assert response.json()["code"] == "INTERNAL_ERROR"
        assert "sensitive" not in response.text + caplog.text


@pytest.mark.parametrize("document_type", [
    "PAYROLL_REPORT", "EXPENSE_CLAIM", "CREDIT_NOTE", "SALES_REPORT",
    "SETTLEMENT_REPORT", "FX_ADVICE", "PROGRESS_CLAIM", "PAYMENT_CERTIFICATE",
    "OPEN_ITEMS_REGISTER", "LOAN_STATEMENT", "PURCHASE_INVOICE",
])
def test_specific_document_types_do_not_match_generic_categories(document, monkeypatch, document_type):
    from app.classification import classify
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(settings, "classification_provider", "MOCK")
    monkeypatch.setattr(settings, "environment", "development")
    types = ["BANK_STATEMENT", "SALES_INVOICE", "PAYMENT_PLATFORM_REPORT", document_type]
    requirements = [{"id": uuid4(), "document_type": value, "title": value} for value in types]
    body = AnalyzeRequest(run_id=uuid4(), purpose="CLASSIFY", documents=[document.model_copy(update={"original_name": document_type.lower() + ".pdf"})], requirements=requirements)
    result = classify(body, f"{body.run_id}:0").classifications[0]
    assert result.document_type == document_type
    assert result.requirement_id == requirements[-1]["id"]


def test_mock_classification_reads_full_file_and_replays(document, monkeypatch):
    from app.classification import classify
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(settings, "classification_provider", "MOCK")
    monkeypatch.setattr(settings, "environment", "development")
    content = b"%PDF-1.7\n" + b"x" * 10000 + b" bank statement"
    (settings.document_path / "sample").write_bytes(content)
    document = document.model_copy(update={"sha256": sha256(content).hexdigest()})
    req_id = uuid4()
    body = AnalyzeRequest(run_id=uuid4(), purpose="CLASSIFY", documents=[document], requirements=[{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Bank"}])
    key = f"{body.run_id}:0"
    output = classify(body, key)
    assert output.classifications[0].requirement_id == req_id
    assert output == classify(body, key)
    assert set(output.model_dump()) == {"schema_version", "run_id", "model_version", "classifications"}
    with pytest.raises(AgentError, match="different request"):
        classify(body.model_copy(update={"run_id": uuid4()}), key)
    monkeypatch.setattr(settings, "environment", "production")
    body = body.model_copy(update={"run_id": uuid4()})
    with pytest.raises(AgentError) as caught:
        classify(body, f"{body.run_id}:0")
    assert caught.value.code == "MOCK_NOT_ALLOWED"


@pytest.mark.parametrize("invalid", [None, "run", "requirement", "order", "finding", "confidence", "timeout"])
def test_remote_classification_boundary(document, monkeypatch, invalid):
    import base64
    import json
    from app import classification
    from app.schemas import AnalyzeRequest

    monkeypatch.setattr(settings, "classification_provider", "REMOTE")
    monkeypatch.setattr(settings, "model_api_url", "https://model.test/classify")
    monkeypatch.setattr(settings, "model_api_key", SecretStr("secret-key"))
    req_id = uuid4()
    body = AnalyzeRequest(run_id=uuid4(), purpose="CLASSIFY", documents=[document], requirements=[{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Bank"}])
    key = f"{body.run_id}:0"

    def respond(request):
        payload = json.loads(request.content)
        assert base64.b64decode(payload["documents"][0]["content_base64"]) == (settings.document_path / "sample").read_bytes()
        assert "storage_key" not in payload["documents"][0]
        assert request.headers["Idempotency-Key"] == key
        assert request.headers["Authorization"] == "Bearer secret-key"
        if invalid == "timeout":
            raise httpx.ReadTimeout("private remote content")
        result = {"schema_version": "1", "run_id": str(body.run_id), "model_version": "test-v1", "classifications": [{"document_id": str(document.document_id), "category": "REQUIREMENT", "requirement_id": str(req_id), "document_type": "BANK_STATEMENT", "confidence": 0.9}]}
        if invalid == "run": result["run_id"] = str(uuid4())
        if invalid == "requirement": result["classifications"][0]["requirement_id"] = str(uuid4())
        if invalid == "order": result["classifications"][0]["document_id"] = str(uuid4())
        if invalid == "finding": result["classifications"][0]["finding"] = "APPROVE"
        if invalid == "confidence": result["classifications"][0]["confidence"] = 2
        return httpx.Response(200, json=result)

    real_client = httpx.Client
    monkeypatch.setattr(classification.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    if invalid:
        with pytest.raises(AgentError) as caught:
            classification.classify(body, key)
        assert caught.value.code == ("MODEL_TIMEOUT" if invalid == "timeout" else "MODEL_INVALID_RESPONSE")
        assert "private" not in str(caught.value)
    else:
        assert classification.classify(body, key).classifications[0].requirement_id == req_id
