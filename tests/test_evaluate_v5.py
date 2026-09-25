import json
from types import SimpleNamespace

import pytest

from app.analysis_schemas import Search
from scripts import evaluate_v5


@pytest.mark.parametrize("document_type,analysis_type", [
    ("BANK_STATEMENT", "BANK_TRANSACTION_RECONCILIATION"),
    ("SUPPLIER_INVOICE", "DOCUMENT_REQUIREMENT_VALIDATION"),
])
def test_case_task_does_not_supply_review_facts(tmp_path, monkeypatch, document_type, analysis_type):
    for folder in ("agent_input", "environment", "document_store/INITIAL", "document_store/CLIENT_HELD"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent_input/task.json").write_text(json.dumps({
        "client_name": "Task name", "reporting_period": "2026-02",
        "instruction": "TASK_ONLY_SENTINEL", "initial_document_refs": ["bank.pdf"],
        "task_type": "BANK_TRANSACTION_RECONCILIATION",
        "target_transaction": {"date": "2026-02-20", "description": "Vendor", "amount": "-SGD 8,498.05"},
    }))
    (tmp_path / "agent_input/client_profile.json").write_text(json.dumps({
        "name": "Registered client", "base_currency": "SGD", "features": {"has_loan": True, "has_employees": True},
        "bank_accounts": [{"bank": "DBS", "last4": "1234"}],
    }))
    (tmp_path / "environment/case_manifest.json").write_text(json.dumps({"documents": [
        {"file": "bank.pdf", "relative_path": "document_store/INITIAL/bank.pdf",
         "document_type": document_type, "visibility": "INITIAL"},
        {"file": "invoice.pdf", "relative_path": "document_store/CLIENT_HELD/invoice.pdf",
         "document_type": "SUPPLIER_INVOICE", "visibility": "CLIENT_HELD"},
    ]}))
    (tmp_path / "environment/ground_truth.json").write_text('{"answer": {}}')
    (tmp_path / "document_store/INITIAL/bank.pdf").write_bytes(b"bank")
    (tmp_path / "document_store/CLIENT_HELD/invoice.pdf").write_bytes(b"invoice")
    seen = []

    def fake_review(body, *_):
        seen.append(body)
        return SimpleNamespace(findings=[], extractions=[], search=None, model_version="test")

    monkeypatch.setattr(evaluate_v5, "review", fake_review)
    evaluate_v5.evaluate(tmp_path, "complete")
    body = seen[0]
    assert body.context.entity_name == "Registered client"
    assert body.context.industry == "OTHER"
    assert body.context.bank_accounts[0].account_last4 == "1234"
    assert body.context.features.has_loan is True
    assert body.review_preference == "STANDARD"
    assert body.requirements[0].instructions == ""
    assert body.requirements[0].title == "Case review"
    assert body.requirements[0].document_type == document_type
    assert body.requirements[0].analysis_type == analysis_type
    assert all(doc.submission_round == 1 and doc.requirement_ids == [body.requirements[0].id]
               for doc in body.documents)
    assert "8,498.05" not in body.model_dump_json()
    assert "TASK_ONLY_SENTINEL" not in body.model_dump_json()


def test_search_result_has_backend_document_metadata(tmp_path, monkeypatch):
    for folder in ("agent_input", "environment", "document_store/INITIAL", "document_store/SEARCHABLE_CURRENT"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent_input/task.json").write_text(json.dumps({
        "reporting_period": "2026-02", "initial_document_refs": ["bank.pdf"],
    }))
    (tmp_path / "agent_input/client_profile.json").write_text(json.dumps({
        "name": "Client", "base_currency": "SGD", "bank_accounts": [],
    }))
    (tmp_path / "environment/case_manifest.json").write_text(json.dumps({"documents": [
        {"file": "bank.pdf", "relative_path": "document_store/INITIAL/bank.pdf",
         "document_type": "BANK_STATEMENT", "visibility": "INITIAL"},
        {"file": "invoice.pdf", "relative_path": "document_store/SEARCHABLE_CURRENT/invoice.pdf",
         "document_type": "SUPPLIER_INVOICE", "visibility": "SEARCHABLE_CURRENT"},
    ]}))
    (tmp_path / "environment/ground_truth.json").write_text('{"answer": {}}')
    (tmp_path / "document_store/INITIAL/bank.pdf").write_bytes(b"bank")
    (tmp_path / "document_store/SEARCHABLE_CURRENT/invoice.pdf").write_bytes(b"invoice")
    seen = []

    def fake_review(body, *_):
        seen.append(body)
        search = Search(action="SEARCH_CURRENT", requirement_id=body.requirements[0].id) if body.turn == 0 else None
        return SimpleNamespace(findings=[], extractions=[], search=search, model_version="test")

    monkeypatch.setattr(evaluate_v5, "review", fake_review)
    evaluate_v5.evaluate(tmp_path, "trajectory")
    assert len(seen) == 2
    assert seen[1].documents[1].submission_round is None
    assert seen[1].documents[1].requirement_ids == []
    assert seen[1].search_history[0].action == "SEARCH_CURRENT"
