"""Backend–Agent REVIEW contract. Keep identical in both repositories."""
from datetime import date
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Money = Annotated[str, Field(pattern=r"^-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,8})?$")]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewFile(StrictModel):
    document_id: UUID
    storage_key: str = Field(min_length=1, max_length=200)
    content_type: Literal["application/pdf", "image/png", "image/jpeg"]
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_name: str = Field(max_length=255)
    requirement_ids: list[UUID] = Field(default_factory=list, max_length=100)
    scope: Literal["CURRENT", "HISTORY"] = "CURRENT"


class ReviewTarget(StrictModel):
    id: UUID
    document_type: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    analysis_type: Literal["DOCUMENT_REQUIREMENT_VALIDATION", "BANK_TRANSACTION_RECONCILIATION"]
    required: bool
    instructions: str = Field(default="", max_length=4000)


class ReviewClientFeatures(StrictModel):
    uses_payment_platform: bool = False
    has_employee_reimbursement: bool = False
    has_loan: bool = False
    multi_currency: bool = False
    project_based: bool = False
    has_retention: bool = False


class ReviewBankAccount(StrictModel):
    bank: str = Field(min_length=1, max_length=100)
    account_last4: str = Field(pattern=r"^[0-9]{4}$")
    currency: Currency


class ReviewContext(StrictModel):
    entity_name: str = Field(min_length=1, max_length=200)
    period: date
    submission_id: UUID
    industry: str | None = Field(default=None, max_length=40)
    base_currency: Currency | None = None
    features: ReviewClientFeatures | None = None
    bank_accounts: list[ReviewBankAccount] = Field(default_factory=list)


class Search(StrictModel):
    action: Literal["SEARCH_CURRENT", "SEARCH_HISTORY"]
    requirement_id: UUID
    document_type: str | None = Field(default=None, max_length=64)
    period: date | None = None
    query: str = Field(default="", max_length=200)
    amount: Money | None = None
    currency: Currency | None = None


class ReviewRequest(StrictModel):
    schema_version: Literal["1"] = "1"
    run_id: UUID
    purpose: Literal["REVIEW"]
    turn: int = Field(default=0, ge=0, le=3)
    context: ReviewContext
    documents: list[ReviewFile] = Field(max_length=100)
    requirements: list[ReviewTarget] = Field(max_length=100)
    search_history: list[Search] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def valid_ids(self):
        ids = {r.id for r in self.requirements}
        if len(ids) != len(self.requirements) or len({d.document_id for d in self.documents}) != len(self.documents):
            raise ValueError("Duplicate IDs")
        if any(not set(d.requirement_ids) <= ids for d in self.documents):
            raise ValueError("Unknown requirement")
        if len(self.search_history) != self.turn:
            raise ValueError("Invalid search turn")
        return self


class Transaction(StrictModel):
    date: date
    description: str = Field(max_length=500)
    amount: Money
    currency: Currency


class Extraction(StrictModel):
    document_id: UUID
    document_type: str | None = Field(default=None, max_length=64)
    entity_name: str | None = Field(default=None, max_length=200)
    period: date | None = None
    invoice_number: str | None = Field(default=None, max_length=100)
    counterparty: str | None = Field(default=None, max_length=200)
    amount: Money | None = None
    currency: Currency | None = None
    transactions: list[Transaction] = Field(default_factory=list, max_length=500)


class Evidence(StrictModel):
    document_id: UUID
    relation: Literal["SUPPORTS", "CONTRADICTS", "REFERENCE"]
    reason: str = Field(min_length=1, max_length=1000)


class AmountOperand(StrictModel):
    document_id: UUID
    amount: Money
    label: str = Field(min_length=1, max_length=200)


class AmountRelation(StrictModel):
    currency: Currency
    operation: Literal["SUM", "SUBTRACT", "MULTIPLY"]
    operands: list[AmountOperand] = Field(min_length=1, max_length=30)
    expected_amount: Money
    actual_amount: Money
    difference: Money


class Finding(StrictModel):
    requirement_id: UUID
    action: Literal["ASK_CLIENT", "RESOLVE", "ESCALATE"]
    suggested_decision: Literal["SATISFY", "REQUEST_ACTION"] | None
    issue_code: Literal["MISSING", "WRONG_PERIOD", "ENTITY_MISMATCH", "UNREADABLE", "INCOMPLETE", "OTHER"] | None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    entity_check: Literal["MATCH", "MISMATCH", "UNKNOWN"]
    period_check: Literal["MATCH", "MISMATCH", "UNKNOWN"]
    explanation: str = Field(min_length=1, max_length=2000)
    client_message: str | None = Field(default=None, max_length=2000)
    evidence: list[Evidence] = Field(max_length=100)
    amounts: list[AmountRelation] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def valid_action(self):
        if self.action == "ASK_CLIENT" and (self.suggested_decision != "REQUEST_ACTION" or not self.issue_code or not self.client_message):
            raise ValueError("Client action requires a reason and message")
        if self.action == "RESOLVE" and (self.suggested_decision != "SATISFY" or not self.evidence or self.issue_code):
            raise ValueError("Resolution requires supporting evidence")
        if self.action == "ESCALATE" and self.suggested_decision is not None:
            raise ValueError("Escalation is not a decision")
        ids = [e.document_id for e in self.evidence]
        if len(ids) != len(set(ids)) or any(o.document_id not in ids for a in self.amounts for o in a.operands):
            raise ValueError("Invalid amount evidence")
        return self


class ReviewResponse(StrictModel):
    schema_version: Literal["1"]
    run_id: UUID
    model_version: str = Field(min_length=1, max_length=200)
    extractions: list[Extraction] = Field(max_length=100)
    findings: list[Finding] = Field(max_length=100)
    search: Search | None = None


def validate_review(body: ReviewRequest, result: object) -> ReviewResponse:
    output = ReviewResponse.model_validate(result)
    req_ids = {r.id for r in body.requirements}
    doc_ids = {d.document_id for d in body.documents}
    finding_ids = [f.requirement_id for f in output.findings]
    extracted_ids = [e.document_id for e in output.extractions]
    if output.run_id != body.run_id or len(set(finding_ids)) != len(finding_ids) or not set(finding_ids) <= req_ids:
        raise ValueError("Unknown or duplicate finding")
    if len(set(extracted_ids)) != len(extracted_ids) or not set(extracted_ids) <= doc_ids:
        raise ValueError("Unknown or duplicate extraction")
    if any(e.document_id not in doc_ids for f in output.findings for e in f.evidence):
        raise ValueError("Unknown evidence")
    if output.search:
        if output.search.requirement_id not in req_ids or output.findings:
            raise ValueError("Search response must not contain final findings")
    elif set(finding_ids) != req_ids or set(extracted_ids) != doc_ids:
        raise ValueError("Final review must cover every input requirement and document")
    return output
