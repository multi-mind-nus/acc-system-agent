import os
import stat
from hashlib import sha256

from app.config import Settings
from app.errors import AgentError
from app.schemas import DocumentReference

SIGNATURES = {
    "application/pdf": b"%PDF-",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}


def read_document(document: DocumentReference, settings: Settings) -> bytes:
    parts = document.storage_key.split("/")
    if any(part in ("", ".", "..") for part in parts) or "\\" in document.storage_key or "\0" in document.storage_key:
        raise AgentError(422, "INVALID_STORAGE_KEY", "Invalid document reference")
    # Open relative to directory descriptors; reject symlinks at every component.
    # This also prevents a path swap between validation and opening the file.
    directory = None
    descriptor = None
    try:
        directory = os.open(settings.document_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in parts[:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise AgentError(422, "INVALID_DOCUMENT", "Document must be a regular file")
        if not 0 < info.st_size <= settings.max_document_bytes:
            raise AgentError(413, "DOCUMENT_SIZE_INVALID", "Document size is outside the allowed range")
        with os.fdopen(descriptor, "rb") as source:
            descriptor = None
            data = source.read(settings.max_document_bytes + 1)
        if len(data) > settings.max_document_bytes:
            raise AgentError(413, "DOCUMENT_SIZE_INVALID", "Document is too large")
        if sha256(data).hexdigest() != document.sha256:
            raise AgentError(422, "DOCUMENT_HASH_MISMATCH", "Document checksum does not match")
        if not data.startswith(SIGNATURES[document.content_type]):
            raise AgentError(415, "DOCUMENT_TYPE_MISMATCH", "Document content type does not match")
        return data
    except OSError:
        raise AgentError(422, "DOCUMENT_UNAVAILABLE", "Document cannot be read safely") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)
