from copy import deepcopy
from uuid import uuid4

import base64
import json
import httpx
import pytest
from fastapi.testclient import TestClient

from app.analysis_schemas import ReviewRequest, validate_review
from app.config import settings
from app.main import app
from app.model_client import analyze_model
from app.config import Settings
from app.review import mock_review
from test_boundary import document  # noqa: F401


def request_body(document, case):
    req = uuid4()
    return ReviewRequest.model_validate({"schema_version": "1", "run_id": str(uuid4()), "purpose": "REVIEW", "context": {"entity_name": "Demo client", "period": "2026-09-01", "submission_id": str(uuid4())},
        "documents": [{**document.model_dump(mode="json"), "original_name": case + ".pdf", "requirement_ids": [str(req)]}],
        "requirements": [{"id": str(req), "document_type": "BANK_STATEMENT", "title": "Bank statement", "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}]})


@pytest.mark.parametrize("case,action", [("G02", "ASK_CLIENT"), ("G03", "ASK_CLIENT"), ("B01", "RESOLVE"), ("B03", "SEARCH_HISTORY"), ("F02", "RESOLVE"), ("F04", "RESOLVE"), ("F05", "RESOLVE"), ("ordinary", "ESCALATE")])
def test_review_cases_and_idempotency(document, monkeypatch, case, action):
    monkeypatch.setattr(settings, "review_provider", "MOCK")
    body = request_body(document, case)
    with TestClient(app) as client:
        headers = {"Idempotency-Key": f"{body.run_id}:0"}
        first = client.post("/v1/analyze", json=body.model_dump(mode="json"), headers=headers)
        assert first.status_code == 200, first.text
        output = validate_review(body, first.json())
        assert output.model_version == "mock-reviewer-v1"
        assert (output.search.action if output.search else output.findings[0].action) == action
        assert client.post("/v1/analyze", json=body.model_dump(mode="json"), headers=headers).json() == first.json()
        changed = body.model_dump(mode="json")
        changed["context"]["entity_name"] = "Different client"
        assert client.post("/v1/analyze", json=changed, headers=headers).status_code == 409
        monkeypatch.setattr(settings, "environment", "production")
        assert client.post("/v1/analyze", json=body.model_dump(mode="json"), headers=headers).json()["code"] == "MOCK_NOT_ALLOWED"


def test_review_rejects_invalid_output_and_search_round(document):
    body = request_body(document, "B01")
    valid = mock_review(body)
    variants = []
    for key, value in (("run_id", str(uuid4())), ("secret", "unexpected")):
        output = deepcopy(valid)
        output[key] = value
        variants.append(output)
    for field, value in (("action", "APPROVE"), ("issue_code", "UNRECOGNIZED"), ("confidence", 2), ("suggested_decision", "WAIVE")):
        output = deepcopy(valid)
        output["findings"][0][field] = value
        variants.append(output)
    output = deepcopy(valid)
    output["findings"][0]["evidence"][0]["document_id"] = str(uuid4())
    variants.append(output)
    output = deepcopy(valid)
    output["findings"][0]["amounts"][0]["actual_amount"] = 2828.80
    variants.append(output)
    for output in variants:
        with pytest.raises(ValueError):
            validate_review(body, output)
    with pytest.raises(ValueError):
        ReviewRequest.model_validate({**body.model_dump(mode="json"), "turn": 1})


def test_remote_review_sends_complete_files_without_storage_paths(document, monkeypatch):
    body = request_body(document, "B01")
    content = b"%PDF-1.7\ncomplete sample"
    def respond(request):
        payload = json.loads(request.content)
        assert payload["purpose"] == "REVIEW"
        assert "storage_key" not in payload["documents"][0]
        assert base64.b64decode(payload["documents"][0]["content_base64"]) == content
        assert request.headers["Idempotency-Key"] == f"{body.run_id}:0"
        return httpx.Response(200, json=mock_review(body))
    real_client = httpx.Client
    monkeypatch.setattr("app.model_client.httpx.Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    configured = Settings(model_api_url="https://model.test/analyze", model_api_key="test-only-key")
    result = analyze_model(body, [content], f"{body.run_id}:0", configured)
    assert validate_review(body, result).findings[0].action == "RESOLVE"
