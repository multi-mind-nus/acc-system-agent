from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: UUID
    storage_key: str = Field(min_length=1, max_length=200)
    content_type: Literal["application/pdf", "image/png", "image/jpeg"]
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_name: str = Field(default="", max_length=255)


class ClassificationRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    document_type: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)


class AnalyzeRequest(BaseModel):
    """Classification accepts no review context or extraction fields."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: UUID
    purpose: Literal["CLASSIFY"]
    documents: list[DocumentReference] = Field(min_length=1, max_length=100)
    requirements: list[ClassificationRequirement] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def unique_documents(self):
        if len({doc.document_id for doc in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique")
        if len({req.id for req in self.requirements}) != len(self.requirements):
            raise ValueError("Requirement IDs must be unique")
        return self


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: UUID
    category: Literal["REQUIREMENT", "OTHER", "INVALID"]
    document_type: str | None = Field(max_length=64)
    requirement_id: UUID | None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_target(self):
        if (self.category == "REQUIREMENT") != (self.requirement_id is not None):
            raise ValueError("Invalid classification target")
        return self


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"]
    run_id: UUID
    model_version: str = Field(min_length=1, max_length=200)
    classifications: list[Classification]
