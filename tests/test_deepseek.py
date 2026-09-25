import base64
import io
import json
from hashlib import sha256
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from app import main
from app import deepseek, ocr
from app.analysis_schemas import ReviewRequest
from app.config import settings


@pytest.mark.parametrize("content_type", ["image/png", "application/pdf"])
def test_ocr_precedes_flash_and_inline_api_is_standalone(tmp_path, monkeypatch, content_type):
    image = Image.new("RGB", (100, 100), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG" if content_type == "image/png" else "PDF")
    content = buffer.getvalue()
    req_id, run_id, doc_id = (str(uuid4()) for _ in range(3))
    monkeypatch.setattr(settings, "classification_provider", "DEEPSEEK")
    monkeypatch.setattr(settings, "model_name", "deepseek/test-flash")
    monkeypatch.setattr(settings, "model_api_key", SecretStr("flash-secret"))
    monkeypatch.setattr(settings, "ocr_api_url", "https://ocr.test/v1/chat/completions")
    monkeypatch.setattr(settings, "ocr_api_key", SecretStr("ocr-secret"))
    monkeypatch.setattr(settings, "agent_api_key", SecretStr("agent-secret"))
    monkeypatch.setattr(settings, "document_path", tmp_path / "absent")
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        if request.url.host == "ocr.test":
            calls.append("ocr")
            assert request.headers["Authorization"] == "Bearer ocr-secret"
            assert payload["model"] == "deepseek/deepseek-ocr-2"
            assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "BANK STATEMENT UOB"}}]})
        calls.append("flash")
        assert payload["model"] == "deepseek/test-flash"
        assert payload["max_tokens"] == 16384
        assert request.headers["Authorization"] == "Bearer flash-secret"
        assert "BANK STATEMENT UOB" in payload["messages"][1]["content"]
        assert "content_base64" not in payload["messages"][1]["content"]
        result = {"schema_version": "1", "run_id": run_id, "model_version": "gpt-4o", "classifications": [
            {"document_id": doc_id, "category": "REQUIREMENT", "document_type": "BANK_STATEMENT", "requirement_id": req_id, "confidence": 0.9}]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    body = {"run_id": run_id, "purpose": "CLASSIFY", "documents": [{"document_id": doc_id, "content_base64": base64.b64encode(content).decode(),
            "content_type": content_type, "sha256": sha256(content).hexdigest(), "original_name": "bank.pdf"}],
            "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Bank"}]}
    with TestClient(main.app) as client:
        headers = {"Idempotency-Key": f"{run_id}:0", "Authorization": "Bearer agent-secret"}
        assert client.post("/v1/analyze-inline", json=body, headers={"Idempotency-Key": f"{run_id}:0"}).status_code == 401
        response = client.post("/v1/analyze-inline", json=body, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["model_version"] == "deepseek/test-flash"
        assert response.json()["classifications"][0]["requirement_id"] == req_id
        assert client.post("/v1/analyze-inline", json=body, headers=headers).json() == response.json()
        body["documents"][0]["sha256"] = "0" * 64
        assert client.post("/v1/analyze-inline", json=body, headers=headers).json()["code"] == "DOCUMENT_HASH_MISMATCH"
        body["documents"][0]["sha256"] = sha256(content).hexdigest()
        body["requirements"].append(body["requirements"][0])
        assert client.post("/v1/analyze-inline", json=body, headers=headers).json()["code"] == "VALIDATION_ERROR"
    assert calls == (["ocr", "flash"] if content_type == "image/png" else ["ocr", "flash"])


def test_review_uses_ocr_and_cannot_create_business_decisions(tmp_path, monkeypatch):
    image = Image.new("RGB", (80, 80), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    run_id, doc_id, req_id, submission_id = (str(uuid4()) for _ in range(4))
    monkeypatch.setattr(settings, "review_provider", "DEEPSEEK")
    monkeypatch.setattr(settings, "model_name", "deepseek/test-flash")
    monkeypatch.setattr(settings, "ocr_api_url", "https://ocr.test/v1/chat/completions")
    monkeypatch.setattr(settings, "novita_api_key", SecretStr("shared-secret"))
    monkeypatch.setattr(settings, "agent_api_key", SecretStr("agent-secret"))
    monkeypatch.setattr(settings, "document_path", tmp_path / "absent")

    def respond(request):
        assert request.headers["Authorization"] == "Bearer shared-secret"
        if request.url.host == "ocr.test":
            assert json.loads(request.content)["model"] == "deepseek/deepseek-ocr-2"
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Invoice period August 2026"}}]})
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek/test-flash"
        prompt = payload["messages"][1]["content"]
        assert "Invoice period August 2026" in prompt
        assert json.loads(prompt)["documents"][0]["submission_round"] == 1
        assert "original_name" not in json.loads(prompt)["documents"][0]
        assert "content_base64" not in prompt
        result = {"schema_version": "1", "run_id": run_id, "model_version": "gpt-4",
                  "extractions": [{"document_id": doc_id, "period": "2026-08-01"}],
                  "findings": [{"requirement_id": req_id, "action": "ESCALATE", "suggested_decision": None,
                                "issue_code": None, "entity_check": "UNKNOWN", "period_check": "UNKNOWN",
                                "explanation": "The OCR text is insufficient to verify the entity.", "evidence": [], "amounts": []}]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    body = {"run_id": run_id, "purpose": "REVIEW", "context": {"entity_name": "Demo", "period": "2026-08-01", "submission_id": submission_id},
            "documents": [{"document_id": doc_id, "content_base64": base64.b64encode(content).decode(), "content_type": "image/png",
                           "sha256": sha256(content).hexdigest(), "original_name": "invoice.png",
                           "submission_round": 1, "requirement_ids": [req_id]}],
            "requirements": [{"id": req_id, "document_type": "SALES_INVOICE", "title": "Invoice",
                              "analysis_type": "DOCUMENT_REQUIREMENT_VALIDATION", "required": True}]}
    with TestClient(main.app) as client:
        response = client.post("/v1/analyze-inline", json=body, headers={"Authorization": "Bearer agent-secret", "Idempotency-Key": f"{run_id}:0"})
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == "deepseek/test-flash"
    assert response.json()["findings"][0]["action"] == "ESCALATE"
    assert "decision" not in response.json()


def test_novita_default_readiness_uses_one_secret_for_both_models(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "document_path", tmp_path)
    monkeypatch.setattr(settings, "classification_provider", "DEEPSEEK")
    monkeypatch.setattr(settings, "review_provider", "DEEPSEEK")
    monkeypatch.setattr(settings, "novita_api_key", SecretStr("shared-secret"))
    seen = []

    def respond(request):
        seen.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer shared-secret"
        return httpx.Response(200, json={"data": []})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    with TestClient(main.app) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200, response.text
    assert seen == ["https://api.novita.ai/openai/v1/models"] * 2


def test_repeated_document_uses_successful_ocr_cache(monkeypatch):
    image = Image.new("RGB", (40, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    monkeypatch.setattr(settings, "ocr_api_url", "https://ocr.test/v1/chat/completions")
    monkeypatch.setattr(settings, "ocr_api_key", SecretStr("ocr-secret"))
    ocr._cache.clear()
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "Invoice 100 SGD"}}]})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    assert ocr.ocr_document(buffer.getvalue(), "image/png", settings) == "[page 1]\nInvoice 100 SGD"
    assert ocr.ocr_document(buffer.getvalue(), "image/png", settings) == "[page 1]\nInvoice 100 SGD"
    assert len(calls) == 1


@pytest.mark.parametrize("bad_relation", [None, "wrong_arithmetic", "missing_conversion", "unreconciled"])
def test_reconciliation_retries_unproved_or_invalid_amounts(monkeypatch, bad_relation):
    run_id, req_id, bank_id, invoice_id = (str(uuid4()) for _ in range(4))
    body = ReviewRequest.model_validate({
        "run_id": run_id, "purpose": "REVIEW",
        "context": {"entity_name": "Demo", "period": "2026-08-01", "submission_id": str(uuid4()), "base_currency": "SGD"},
        "documents": [
            {"document_id": bank_id, "storage_key": "bank.pdf", "content_type": "application/pdf", "sha256": "0" * 64,
             "original_name": "bank.pdf", "requirement_ids": [req_id]},
            {"document_id": invoice_id, "storage_key": "invoice.pdf", "content_type": "application/pdf", "sha256": "0" * 64,
             "original_name": "invoice.pdf", "requirement_ids": [req_id]},
        ],
        "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Reconcile payment",
                          "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}],
    })
    relation = {"currency": "SGD", "operation": "SUM", "operands": [
        {"document_id": invoice_id, "amount": "100.00", "label": "Invoice"},
        {"document_id": bank_id, "amount": "-100.00", "label": "Bank debit"}],
        "expected_amount": "0", "actual_amount": "0", "difference": "0"}
    conversion = {"currency": "SGD", "operation": "MULTIPLY", "operands": [
        {"document_id": invoice_id, "amount": "100.00", "label": "Foreign amount"},
        {"document_id": invoice_id, "amount": "1.00", "label": "Exchange rate"}],
        "expected_amount": "100.00", "actual_amount": "100.00", "difference": "0"}
    finding = {"requirement_id": req_id, "action": "RESOLVE", "suggested_decision": "SATISFY", "issue_code": None,
               "entity_check": "MATCH", "period_check": "MATCH", "explanation": "Payment matches invoice.",
               "evidence": [{"document_id": bank_id, "relation": "SUPPORTS", "reason": "Payment"},
                            {"document_id": invoice_id, "relation": "SUPPORTS", "reason": "Invoice"}], "amounts": [relation]}
    responses = []
    for attempt in range(2):
        item = dict(finding)
        if attempt == 0:
            item["amounts"] = ([] if bad_relation is None else
                               [relation] if bad_relation == "missing_conversion" else
                               [{**relation, "expected_amount": "-10", "difference": "10"}] if bad_relation == "unreconciled" else
                               [{**relation, "actual_amount": "1"}])
        elif bad_relation == "missing_conversion":
            item["amounts"] = [conversion, relation]
        responses.append({"schema_version": "1", "run_id": run_id, "model_version": "deepseek/test-flash",
                          "extractions": [{"document_id": bank_id}, {"document_id": invoice_id,
                                           "currency": "EUR" if bad_relation == "missing_conversion" else "SGD"}],
                          "findings": [item]})
    prompts = []
    monkeypatch.setattr(deepseek, "ocr_document", lambda *args: "Bank debit SGD 100; invoice SGD 100")

    def fake_flash(prompt, payload, config, deadline):
        prompts.append(prompt)
        return responses[len(prompts) - 1]

    monkeypatch.setattr(deepseek, "_flash", fake_flash)
    result = deepseek.review_with_deepseek(body, [b"bank", b"invoice"], settings)
    assert result["findings"][0]["amounts"] == ([conversion, relation] if bad_relation == "missing_conversion" else [relation])
    assert len(prompts) == 2 and "failed validation" in prompts[1]


def test_entity_mismatch_cannot_contradict_cited_ocr(monkeypatch):
    run_id, req_id, doc_id = (str(uuid4()) for _ in range(3))
    body = ReviewRequest.model_validate({
        "run_id": run_id, "purpose": "REVIEW",
        "context": {"entity_name": "Demo Pte. Ltd.", "period": "2026-08-01", "submission_id": str(uuid4())},
        "documents": [{"document_id": doc_id, "storage_key": "bank.pdf", "content_type": "application/pdf",
                       "sha256": "0" * 64, "original_name": "bank.pdf", "requirement_ids": [req_id]}],
        "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Reconcile payment",
                          "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}],
    })
    wrong = {"schema_version": "1", "run_id": run_id, "model_version": "deepseek/test-flash",
             "extractions": [{"document_id": doc_id}], "findings": [{
                 "requirement_id": req_id, "action": "ASK_CLIENT", "suggested_decision": "REQUEST_ACTION",
                 "issue_code": "ENTITY_MISMATCH", "entity_check": "MISMATCH",
                 "period_check": "MATCH", "explanation": "Wrong holder", "client_message": "Send correct statement",
                 "evidence": [{"document_id": doc_id, "relation": "CONTRADICTS", "reason": "Holder"}]}]}
    corrected = {"schema_version": "1", "run_id": run_id, "model_version": "deepseek/test-flash",
                 "extractions": [], "findings": [], "search": {"action": "SEARCH_CURRENT", "requirement_id": req_id}}
    responses = iter((wrong, corrected))
    monkeypatch.setattr(deepseek, "ocr_document", lambda *args: "Account Holder: Demo Pte. Ltd.\nPayee: Other Vendor")
    monkeypatch.setattr(deepseek, "_flash", lambda *args: next(responses))
    assert deepseek.review_with_deepseek(body, [b"bank"], settings)["search"]["action"] == "SEARCH_CURRENT"


@pytest.mark.parametrize("searched,expected", [([], "SEARCH_CURRENT"), (["SEARCH_CURRENT"], "SEARCH_HISTORY"),
                                             (["SEARCH_CURRENT", "SEARCH_HISTORY"], None)])
def test_missing_bank_support_searches_before_escalation(monkeypatch, searched, expected):
    run_id, req_id, doc_id = (str(uuid4()) for _ in range(3))
    body = ReviewRequest.model_validate({
        "run_id": run_id, "purpose": "REVIEW", "turn": len(searched),
        "context": {"entity_name": "Demo", "period": "2026-08-01", "submission_id": str(uuid4())},
        "documents": [{"document_id": doc_id, "storage_key": "bank.pdf", "content_type": "application/pdf",
                       "sha256": "0" * 64, "original_name": "bank.pdf", "requirement_ids": [req_id]}],
        "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Reconcile payment",
                          "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}],
        "search_history": [{"action": action, "requirement_id": req_id} for action in searched],
    })
    response = {"schema_version": "1", "run_id": run_id, "model_version": "deepseek/test-flash",
                "extractions": [{"document_id": doc_id}], "findings": [{
                    "requirement_id": req_id, "action": "ESCALATE", "suggested_decision": None,
                    "issue_code": "INCOMPLETE", "entity_check": "MATCH", "period_check": "MATCH",
                    "explanation": "Only the bank statement is available.",
                    "evidence": [{"document_id": doc_id, "relation": "REFERENCE", "reason": "Bank entry"}]}]}
    monkeypatch.setattr(deepseek, "ocr_document", lambda *args: "Bank payment SGD 100")
    monkeypatch.setattr(deepseek, "_flash", lambda *args: response)
    result = deepseek.review_with_deepseek(body, [b"bank"], settings)
    assert (result.get("search") or {}).get("action") == expected


def test_register_alone_cannot_prove_bank_payment(monkeypatch):
    run_id, req_id, bank_id, register_id = (str(uuid4()) for _ in range(4))
    body = ReviewRequest.model_validate({
        "run_id": run_id, "purpose": "REVIEW",
        "context": {"entity_name": "Demo", "period": "2026-08-01", "submission_id": str(uuid4())},
        "documents": [
            {"document_id": bank_id, "storage_key": "bank.pdf", "content_type": "application/pdf",
             "document_type": "BANK_STATEMENT", "sha256": "0" * 64, "original_name": "bank.pdf", "requirement_ids": [req_id]},
            {"document_id": register_id, "storage_key": "register.pdf", "content_type": "application/pdf",
             "document_type": "OPEN_ITEMS_REGISTER", "sha256": "0" * 64, "original_name": "register.pdf", "requirement_ids": [req_id]},
        ],
        "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Reconcile payment",
                          "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}],
    })
    evidence = [{"document_id": bank_id, "relation": "SUPPORTS", "reason": "Debit"},
                {"document_id": register_id, "relation": "SUPPORTS", "reason": "Open item"}]
    base = {"schema_version": "1", "run_id": run_id, "model_version": "deepseek/test-flash",
            "extractions": [{"document_id": bank_id}, {"document_id": register_id}]}
    wrong = {**base, "findings": [{"requirement_id": req_id, "action": "RESOLVE", "suggested_decision": "SATISFY",
                                    "issue_code": None, "entity_check": "MATCH", "period_check": "MATCH",
                                    "explanation": "Register matches debit", "evidence": evidence,
                                    "amounts": [{"currency": "SGD", "operation": "SUM", "operands": [
                                        {"document_id": register_id, "amount": "100", "label": "Open item"},
                                        {"document_id": bank_id, "amount": "-100", "label": "Debit"}],
                                        "expected_amount": "0", "actual_amount": "0", "difference": "0"}]}]}
    corrected = {**base, "findings": [{"requirement_id": req_id, "action": "ASK_CLIENT",
                                        "suggested_decision": "REQUEST_ACTION", "issue_code": "INCOMPLETE",
                                        "entity_check": "MATCH", "period_check": "MATCH",
                                        "explanation": "Source invoice is absent", "client_message": "Provide the source invoice",
                                        "evidence": evidence}]}
    responses = iter((wrong, corrected))
    monkeypatch.setattr(deepseek, "ocr_document", lambda *args: "Bank debit and open item SGD 100")
    monkeypatch.setattr(deepseek, "_flash", lambda *args: next(responses))
    assert deepseek.review_with_deepseek(body, [b"bank", b"register"], settings)["search"]["action"] == "SEARCH_CURRENT"


def test_matching_open_items_register_is_required_in_historical_evidence(monkeypatch):
    run_id, req_id, bank_id, invoice_id, register_id = (str(uuid4()) for _ in range(5))
    body = ReviewRequest.model_validate({
        "run_id": run_id, "purpose": "REVIEW",
        "context": {"entity_name": "Demo", "period": "2026-06-01", "submission_id": str(uuid4())},
        "documents": [
            {"document_id": bank_id, "storage_key": "bank.pdf", "content_type": "application/pdf",
             "document_type": "BANK_STATEMENT", "sha256": "0" * 64, "original_name": "bank.pdf", "requirement_ids": [req_id]},
            {"document_id": invoice_id, "storage_key": "invoice.pdf", "content_type": "application/pdf",
             "document_type": "SUPPLIER_INVOICE", "sha256": "0" * 64, "original_name": "invoice.pdf", "requirement_ids": [req_id]},
            {"document_id": register_id, "storage_key": "register.pdf", "content_type": "application/pdf",
             "document_type": "OPEN_ITEMS_REGISTER", "sha256": "0" * 64, "original_name": "register.pdf", "requirement_ids": [req_id]},
        ],
        "requirements": [{"id": req_id, "document_type": "BANK_STATEMENT", "title": "Reconcile payment",
                          "analysis_type": "BANK_TRANSACTION_RECONCILIATION", "required": True}],
    })
    evidence = [{"document_id": bank_id, "relation": "SUPPORTS", "reason": "Payment"},
                {"document_id": invoice_id, "relation": "SUPPORTS", "reason": "Invoice"}]
    base = {"schema_version": "1", "run_id": run_id, "model_version": "ignored",
            "extractions": [{"document_id": bank_id}, {"document_id": invoice_id, "invoice_number": "INV-1234"},
                            {"document_id": register_id}]}
    finding = {"requirement_id": req_id, "action": "RESOLVE", "suggested_decision": "SATISFY",
               "issue_code": None, "entity_check": "MATCH", "period_check": "MATCH",
               "explanation": "Invoice paid", "evidence": evidence,
               "amounts": [{"currency": "SGD", "operation": "SUM", "operands": [
                   {"document_id": bank_id, "amount": "-100", "label": "Payment"},
                   {"document_id": invoice_id, "amount": "100", "label": "Invoice"}],
                   "expected_amount": "0", "actual_amount": "0", "difference": "0"}]}
    responses = iter(({**base, "findings": [finding]}, {**base, "findings": [{**finding, "evidence": evidence + [
        {"document_id": register_id, "relation": "REFERENCE", "reason": "Listed as unpaid"}]}]}))
    texts = {b"bank": "Bank debit 100", b"invoice": "Supplier invoice INV-1234 100",
             b"register": "Open items INV-1234 unpaid"}
    monkeypatch.setattr(deepseek, "ocr_document", lambda content, *_: texts[content])
    monkeypatch.setattr(deepseek, "_flash", lambda *args: next(responses))
    result = deepseek.review_with_deepseek(body, list(texts), settings)
    assert [item["document_id"] for item in result["findings"][0]["evidence"]] == [bank_id, invoice_id, register_id]
