"""Portable API contract: callers send file bytes, without Folio storage access."""

import base64
import binascii
from hashlib import sha256
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.analysis_schemas import ReviewContext, ReviewFile, ReviewRequest, ReviewTarget, Search
from app.config import settings
from app.errors import AgentError
from app.schemas import AnalyzeRequest, ClassificationRequirement, DocumentReference
from app.storage import SIGNATURES


class InlineDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: UUID
    content_base64: str = Field(min_length=1, max_length=35_000_000)
    content_type: Literal["application/pdf", "image/png", "image/jpeg"]
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_name: str = Field(default="", max_length=255)


class InlineReviewDocument(InlineDocument):
    document_type: str | None = Field(default=None, max_length=64)
    submission_round: int | None = Field(default=None, ge=1)
    requirement_ids: list[UUID] = Field(default_factory=list, max_length=100)
    scope: Literal["CURRENT", "HISTORY"] = "CURRENT"


class InlineClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    run_id: UUID
    purpose: Literal["CLASSIFY"]
    documents: list[InlineDocument] = Field(min_length=1, max_length=100)
    requirements: list[ClassificationRequirement] = Field(default_factory=list, max_length=100)


class InlineReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"
    run_id: UUID
    purpose: Literal["REVIEW"]
    review_preference: Literal["CAUTIOUS", "STANDARD", "EFFICIENT"] = "STANDARD"
    turn: int = Field(default=0, ge=0, le=3)
    context: ReviewContext
    documents: list[InlineReviewDocument] = Field(max_length=100)
    requirements: list[ReviewTarget] = Field(max_length=100)
    search_history: list[Search] = Field(default_factory=list, max_length=3)


def decode_inline(body: InlineClassifyRequest | InlineReviewRequest):
    files = []
    total = 0
    for doc in body.documents:
        try:
            content = base64.b64decode(doc.content_base64, validate=True)
        except (binascii.Error, ValueError):
            raise AgentError(422, "INVALID_DOCUMENT", "Document is not valid base64") from None
        total += len(content)
        if not 0 < len(content) <= settings.max_document_bytes or total > 100 * 1024 * 1024:
            raise AgentError(413, "DOCUMENT_SIZE_INVALID", "Document size is outside the allowed range")
        if sha256(content).hexdigest() != doc.sha256:
            raise AgentError(422, "DOCUMENT_HASH_MISMATCH", "Document checksum does not match")
        if not content.startswith(SIGNATURES[doc.content_type]):
            raise AgentError(415, "DOCUMENT_TYPE_MISMATCH", "Document content type does not match")
        files.append(content)
    payload = body.model_dump(mode="json")
    for doc in payload["documents"]:
        doc.pop("content_base64")
        doc["storage_key"] = f"inline/{doc['document_id']}"
    try:
        parsed = AnalyzeRequest.model_validate(payload) if body.purpose == "CLASSIFY" else ReviewRequest.model_validate(payload)
    except ValidationError:
        raise AgentError(422, "VALIDATION_ERROR", "Request validation failed") from None
    return parsed, files
