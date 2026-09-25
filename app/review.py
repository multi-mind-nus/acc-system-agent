"""Analysis only: no database, search execution, or business decisions."""
from collections import OrderedDict
from datetime import timedelta
from hashlib import sha256
from threading import Lock
from time import perf_counter

from app.analysis_schemas import ReviewRequest, validate_review
from app.config import settings
from app.errors import AgentError
from app.model_client import analyze_model
from app.deepseek import review_with_deepseek
from app.storage import read_document
from app.telemetry import elapsed_ms, event

cache = OrderedDict()
lock = Lock()


def mock_review(body: ReviewRequest):
    # Explicit demo scenarios, NOT an OCR engine or a trained model.
    extractions = [{"document_id": str(doc.document_id)} for doc in body.documents]
    findings = []
    for req in body.requirements:
        docs = [d for d in body.documents if req.id in d.requirement_ids]
        names = " ".join(d.original_name.upper() for d in docs)
        historical = [d for d in body.documents if d.scope == "HISTORY"]
        if "B03" in names and not body.turn:
            return {"schema_version": "1", "run_id": str(body.run_id), "model_version": "mock-reviewer-v1", "extractions": [], "findings": [], "search": {"action": "SEARCH_HISTORY", "requirement_id": str(req.id), "query": "B03"}}
        code = "WRONG_PERIOD" if "G02" in names else "ENTITY_MISMATCH" if "G03" in names else "MISSING" if not docs else None
        scenario = next((s for s in ("B01", "B03", "F02", "F04", "F05") if s in names), None)
        if code or scenario:
            for doc in docs:
                extraction = next(e for e in extractions if e["document_id"] == str(doc.document_id))
                extraction.update(document_type=req.document_type, entity_name="Simulated different entity" if code == "ENTITY_MISMATCH" else body.context.entity_name,
                    period=((body.context.period.replace(day=1) - timedelta(days=1)).replace(day=1) if code == "WRONG_PERIOD" else body.context.period).isoformat())
        supports = docs + (historical if scenario == "B03" else [])
        resolved = bool(scenario and (scenario != "B03" or historical))
        if scenario == "B03" and not historical:
            code = "MISSING"
        finding = {"requirement_id": str(req.id), "action": "ASK_CLIENT" if code else "RESOLVE" if resolved else "ESCALATE", "suggested_decision": "REQUEST_ACTION" if code else "SATISFY" if resolved else None, "issue_code": code,
            "entity_check": "MISMATCH" if code == "ENTITY_MISMATCH" else "UNKNOWN", "period_check": "MISMATCH" if code == "WRONG_PERIOD" else "UNKNOWN",
            "explanation": {"WRONG_PERIOD": "The simulated document period differs from the requested period.", "ENTITY_MISMATCH": "The simulated document belongs to a different entity.", "MISSING": "Supporting documents are missing in this simulation."}.get(code, "Simulated supporting evidence found; verify the source documents." if resolved else "No model connected; manual verification required."),
            "client_message": "Please provide documents for the requested period." if code == "WRONG_PERIOD" else "Please provide documents for the correct entity." if code == "ENTITY_MISMATCH" else "Please supply the missing supporting documents." if code else None,
            "evidence": [{"document_id": str(d.document_id), "relation": "CONTRADICTS" if code else "SUPPORTS", "reason": "Simulated evidence; verify the original document."} for d in supports], "amounts": []}
        if scenario in ("B01", "F02", "F04", "F05") and docs:
            operands, expected = (("1246.02", "1582.78"), "2828.80") if scenario == "B01" else (("120.00", "80.00"), "200.00") if scenario == "F02" else (("1340.00", "10.00"), "1350.00") if scenario == "F04" else (("1000.00", "100.00"), "900.00")
            finding["amounts"] = [{"currency": "SGD", "operation": "SUBTRACT" if scenario == "F05" else "SUM", "operands": [{"document_id": str(docs[min(i, len(docs)-1)].document_id), "amount": amount, "label": "Simulated operand"} for i, amount in enumerate(operands)], "expected_amount": expected, "actual_amount": expected, "difference": "0.00"}]
            if scenario == "F04":
                finding["amounts"].insert(0, {"currency": "SGD", "operation": "MULTIPLY", "operands": [{"document_id": str(docs[0].document_id), "amount": "1000.00", "label": "Simulated USD principal"}, {"document_id": str(docs[0].document_id), "amount": "1.34", "label": "Simulated SGD/USD rate"}], "expected_amount": "1340.00", "actual_amount": "1340.00", "difference": "0.00"})
            extraction = next(e for e in extractions if e["document_id"] == str(docs[0].document_id))
            extraction.update(amount=expected, currency="SGD", transactions=[{"date": body.context.period.isoformat(), "description": "Simulated payment", "amount": "-" + expected, "currency": "SGD"}] if req.document_type == "BANK_STATEMENT" else [])
        findings.append(finding)
    return {"schema_version": "1", "run_id": str(body.run_id), "model_version": "mock-reviewer-v1", "extractions": extractions, "findings": findings}


def review(body: ReviewRequest, key: str, supplied_files: list[bytes] | None = None):
    if key != f"{body.run_id}:{body.turn}":
        raise AgentError(422, "INVALID_IDEMPOTENCY_KEY", "Key must match run and turn")
    if settings.review_provider == "DISABLED":
        raise AgentError(503, "MODEL_NOT_CONFIGURED", "Review is not enabled")
    if settings.review_provider == "MOCK" and settings.environment == "production":
        raise AgentError(503, "MOCK_NOT_ALLOWED", "Simulated review is disabled in production")
    fingerprint = sha256(body.model_dump_json().encode()).hexdigest()
    # ponytail: bounded per-process replay, Backend persists turns; provider must
    # honour idempotency when running multiple Agent replicas.
    with lock:
        if key in cache:
            previous, result = cache[key]
            if previous != fingerprint:
                raise AgentError(409, "IDEMPOTENCY_CONFLICT", "Key was used with a different request")
            event("analysis_reused", run_id=str(body.run_id), purpose="REVIEW", turn=body.turn)
            return result
        event("analysis_started", run_id=str(body.run_id), purpose="REVIEW", turn=body.turn,
              provider=settings.review_provider, document_count=len(body.documents),
              requirement_count=len(body.requirements))
        files, total = [], 0
        for index, doc in enumerate(body.documents):
            started = perf_counter()
            try:
                content = supplied_files[index] if supplied_files is not None else read_document(doc, settings)
            except AgentError as exc:
                event("document_read", run_id=str(body.run_id), document_id=str(doc.document_id),
                      status="failed", error_code=exc.code, duration_ms=elapsed_ms(started))
                raise
            event("document_read", run_id=str(body.run_id), document_id=str(doc.document_id),
                  status="ok", size_bytes=len(content), duration_ms=elapsed_ms(started))
            total += len(content)
            if total > 100 * 1024 * 1024:
                raise AgentError(413, "BATCH_TOO_LARGE", "Review exceeds the document limit")
            files.append(content)
        if settings.review_provider == "MOCK":
            result = mock_review(body)
        elif settings.review_provider == "DEEPSEEK":
            result = review_with_deepseek(body, files, settings)
        else:
            result = analyze_model(body, files, key, settings)
        try:
            output = validate_review(body, result)
        except ValueError:
            event("review_validation", run_id=str(body.run_id), turn=body.turn, status="failed")
            raise AgentError(502, "MODEL_INVALID_RESPONSE", "Invalid review response") from None
        event("review_validation", run_id=str(body.run_id), turn=body.turn, status="ok")
        if output.search:
            event("review_search", run_id=str(body.run_id), turn=body.turn,
                  action=output.search.action, requirement_id=str(output.search.requirement_id))
        for finding in output.findings:
            event("review_finding", run_id=str(body.run_id), turn=body.turn,
                  requirement_id=str(finding.requirement_id), action=finding.action,
                  issue_code=finding.issue_code, suggested_decision=finding.suggested_decision,
                  evidence_count=len(finding.evidence), amount_relation_count=len(finding.amounts))
        cache[key] = (fingerprint, output)
        if len(cache) > 128:
            cache.popitem(last=False)
        event("analysis_completed", run_id=str(body.run_id), purpose="REVIEW", turn=body.turn,
              status="ok", outcome="search" if output.search else "findings")
        return output
