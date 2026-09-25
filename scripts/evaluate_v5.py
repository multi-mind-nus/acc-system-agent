"""Run dataset documents through the real OCR/review path without importing labels into prompts.

Usage: REVIEW_PROVIDER=DEEPSEEK uv run python -m scripts.evaluate_v5 CASE_DIR ...
"""

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from app.analysis_schemas import ReviewRequest
from app.errors import AgentError
from app.review import review


def evaluate(case_dir: Path, phase: str, include_distractors: bool = True) -> dict:
    task = json.loads((case_dir / "agent_input/task.json").read_text())
    profile = json.loads((case_dir / "agent_input/client_profile.json").read_text())
    manifest = json.loads((case_dir / "environment/case_manifest.json").read_text())
    initial = set(task["initial_document_refs"])
    selected = [doc for doc in manifest["documents"]
                if (phase == "complete" or doc["file"] in initial)
                and (include_distractors or not doc.get("distractor"))]
    requirement_id, run_id = uuid4(), uuid4()
    files, documents, names = [], [], {}

    def add_document(entry):
        content = (case_dir / entry["relative_path"]).read_bytes()
        document_id = uuid4()
        names[str(document_id)] = entry["file"]
        files.append(content)
        submitted = (entry["visibility"] == "INITIAL"
                     or (phase == "complete" and entry["visibility"] != "SEARCHABLE_HISTORY")
                     or (phase == "trajectory" and entry["visibility"] == "CLIENT_HELD"))
        documents.append({
            "document_id": str(document_id),
            "storage_key": f"{case_dir.name}/{entry['file']}",
            "content_type": mimetypes.guess_type(entry["file"])[0],
            "document_type": entry["document_type"],
            "submission_round": (2 if phase == "trajectory" and entry["visibility"] == "CLIENT_HELD" else 1)
                                if submitted else None,
            "sha256": hashlib.sha256(content).hexdigest(),
            "original_name": entry["file"],
            "requirement_ids": [str(requirement_id)] if submitted else [],
            "scope": "HISTORY" if entry["visibility"] == "SEARCHABLE_HISTORY" else "CURRENT",
        })

    for entry in selected:
        add_document(entry)
    document_type = task.get("requirement", {}).get("document_type") or next(
        (entry["document_type"] for entry in manifest["documents"] if entry["file"] in initial), None)
    if not document_type:
        raise ValueError("Case has no requested document type")
    features = {key: value for key, value in profile.get("features", {}).items()
                if key in {"uses_payment_platform", "has_employee_reimbursement", "has_loan", "multi_currency", "project_based", "has_retention"}}
    request = {
        "schema_version": "1", "run_id": str(run_id), "purpose": "REVIEW", "turn": 0,
        "review_preference": "STANDARD",
        "context": {
            "entity_name": profile["name"], "period": task["reporting_period"] + "-01",
            "submission_id": str(uuid4()), "industry": profile.get("industry", "OTHER"),
            "base_currency": profile.get("base_currency"),
            "features": features,
            "bank_accounts": [{"bank": bank["bank"], "account_last4": bank["last4"],
                               "currency": profile["base_currency"]} for bank in profile.get("bank_accounts", [])],
        },
        "documents": documents,
        "requirements": [{
            "id": str(requirement_id),
            "document_type": document_type,
            "title": "Case review",
            "analysis_type": "BANK_TRANSACTION_RECONCILIATION" if document_type == "BANK_STATEMENT" else "DOCUMENT_REQUIREMENT_VALIDATION",
            "required": True, "instructions": "",
        }],
        "search_history": [],
    }
    steps = []
    for _ in range(7 if phase == "trajectory" else 1):
        body = ReviewRequest.model_validate(request)
        step = {"input_files": [names[str(doc.document_id)] for doc in body.documents]}
        try:
            output = review(body, f"{body.run_id}:{body.turn}", files)
            finding = output.findings[0] if output.findings else None
            step.update({
                "action": finding.action if finding else None,
                "decision": finding.suggested_decision if finding else None,
                "issue": finding.issue_code if finding else None,
                "explanation": finding.explanation if finding else None,
                "evidence": [names[str(value.document_id)] for value in finding.evidence] if finding else [],
                "amounts": [value.model_dump(mode="json") for value in finding.amounts] if finding else [],
                "extractions": [{"file": names[str(value.document_id)], "document_type": value.document_type,
                                 "currency": value.currency, "amount": value.amount} for value in output.extractions],
                "search": output.search.model_dump(mode="json") if output.search else None,
                "model_version": output.model_version,
            })
        except AgentError as exc:
            step["error"] = exc.code
            step["error_detail"] = exc.message
            steps.append(step)
            break
        steps.append(step)
        if phase == "trajectory":
            print(json.dumps({"case": case_dir.name, "step": len(steps),
                              **{key: step.get(key) for key in ("action", "issue", "search", "error")}},
                             ensure_ascii=False), flush=True)
        if phase != "trajectory":
            break
        if output.search:
            visibility = "SEARCHABLE_HISTORY" if output.search.action == "SEARCH_HISTORY" else "SEARCHABLE_CURRENT"
            known = set(names.values())
            for entry in manifest["documents"]:
                if entry["visibility"] == visibility and entry["file"] not in known:
                    add_document(entry)
            request["turn"] = body.turn + 1
            request["search_history"].append(output.search.model_dump(mode="json"))
        elif finding and finding.action == "ASK_CLIENT":
            held = [entry for entry in manifest["documents"]
                    if entry["visibility"] == "CLIENT_HELD" and entry["file"] not in names.values()]
            if not held:
                break
            for entry in held:
                add_document(entry)
            request["run_id"] = str(uuid4())
            request["context"]["submission_id"] = str(uuid4())
            request["turn"], request["search_history"] = 0, []
        else:
            break
    summary = {"case": case_dir.name, "phase": phase, **steps[-1]}
    if phase == "trajectory":
        summary["steps"] = [{key: value for key, value in step.items() if key in ("action", "issue", "search", "error", "input_files")}
                            for step in steps]
    truth = json.loads((case_dir / "environment/ground_truth.json").read_text())
    answer = truth.get("answer", {})
    summary["expected_evidence"] = answer.get("evidence_used", [])
    summary["expected_formula"] = answer.get("accounting_relationship", {}).get("formula")
    summary["expected_initial_issues"] = truth.get("initial_exceptions", [])
    if phase != "initial" and summary["expected_evidence"]:
        summary["evidence_match"] = set(summary.get("evidence", [])) == set(summary["expected_evidence"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dirs", nargs="+", type=Path)
    parser.add_argument("--phase", choices=("initial", "complete", "trajectory"), default="complete")
    parser.add_argument("--without-distractors", action="store_true")
    args = parser.parse_args()
    for case_dir in args.case_dirs:
        print(json.dumps({"started": case_dir.name, "phase": args.phase}), flush=True)
        print(json.dumps(evaluate(case_dir, args.phase, not args.without_distractors), ensure_ascii=False), flush=True)
