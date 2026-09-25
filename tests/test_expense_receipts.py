from copy import deepcopy
from uuid import uuid4

import pytest

from app.analysis_schemas import ReviewRequest, validate_review


def test_expense_claim_requires_independent_receipt_for_each_line():
    bank, claim, first, second = (str(uuid4()) for _ in range(4))
    receipt_req, claim_req, bank_req = (str(uuid4()) for _ in range(3))
    run_id = str(uuid4())

    def document(document_id, kind, requirement_id):
        return {"document_id": document_id, "storage_key": document_id,
                "content_type": "application/pdf", "sha256": "0" * 64,
                "original_name": "document.pdf", "document_type": kind,
                "requirement_ids": [requirement_id]}

    request = {"run_id": run_id, "purpose": "REVIEW",
               "context": {"entity_name": "Example Ltd", "period": "2026-05-01", "submission_id": str(uuid4())},
               "documents": [document(bank, "BANK_STATEMENT", bank_req),
                             document(claim, "EXPENSE_CLAIM", claim_req),
                             document(first, "RECEIPT", receipt_req)],
               "requirements": [{"id": requirement_id, "document_type": kind, "title": kind,
                                 "analysis_type": "BANK_TRANSACTION_RECONCILIATION" if kind == "BANK_STATEMENT"
                                 else "DOCUMENT_REQUIREMENT_VALIDATION", "required": True}
                                for requirement_id, kind in ((receipt_req, "RECEIPT"),
                                                             (claim_req, "EXPENSE_CLAIM"),
                                                             (bank_req, "BANK_STATEMENT"))]}

    def finding(requirement_id):
        return {"requirement_id": requirement_id, "action": "RESOLVE", "suggested_decision": "SATISFY",
                "issue_code": None, "entity_check": "MATCH", "period_check": "MATCH",
                "explanation": "Documents reconcile", "evidence": [
                    {"document_id": document_id, "relation": "SUPPORTS", "reason": "Source"}
                    for document_id in (bank, claim, first)], "amounts": []}

    output = {"schema_version": "1", "run_id": run_id, "model_version": "test",
              "extractions": [{"document_id": bank},
                              {"document_id": claim, "document_type": "EXPENSE_CLAIM", "amount": "305.92",
                               "currency": "SGD", "transactions": [
                                   {"date": "2026-05-01", "description": "Expense 1", "amount": "39.52", "currency": "SGD"},
                                   {"date": "2026-05-02", "description": "Expense 2", "amount": "266.40", "currency": "SGD"}]},
                              {"document_id": first, "document_type": "RECEIPT", "amount": "39.52", "currency": "SGD"}],
              "findings": [finding(receipt_req), finding(claim_req), finding(bank_req)]}
    output["findings"][1]["amounts"] = [{"currency": "SGD", "operation": "SUM",
        "operands": [{"document_id": first, "amount": "39.52", "label": "Original receipt"},
                     {"document_id": claim, "amount": "266.40", "label": "Claim line only"}],
        "expected_amount": "305.92", "actual_amount": "305.92", "difference": "0.00"}]

    with pytest.raises(ValueError, match="distinct matching receipt"):
        validate_review(ReviewRequest.model_validate(request), output)

    complete_request = deepcopy(request)
    complete_request["documents"].append(document(second, "RECEIPT", receipt_req))
    complete_output = deepcopy(output)
    complete_output["extractions"].append({"document_id": second, "document_type": "RECEIPT",
                                            "amount": "266.40", "currency": "SGD"})
    for row in complete_output["findings"]:
        row["evidence"].append({"document_id": second, "relation": "SUPPORTS", "reason": "Source"})
    with pytest.raises(ValueError, match="original receipt amounts"):
        validate_review(ReviewRequest.model_validate(complete_request), complete_output)

    complete_output["findings"][1]["amounts"][0]["operands"][1]["document_id"] = second
    complete_output["findings"][1]["amounts"][0]["operands"][1]["amount"] = "266.39"
    with pytest.raises(ValueError, match="original receipt amounts"):
        validate_review(ReviewRequest.model_validate(complete_request), complete_output)
    complete_output["findings"][1]["amounts"][0]["operands"][1]["amount"] = "266.40"
    assert len(validate_review(ReviewRequest.model_validate(complete_request), complete_output).findings) == 3
