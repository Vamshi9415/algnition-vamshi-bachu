"""Upload service for converting ad-platform files into canonical datasets."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile


class UploadProcessingError(RuntimeError):
    """Raised when an uploaded file cannot be processed."""


class _LoaderProtocol(Protocol):
    def load(self, path: str):
        """Load a file from the provided filesystem path."""


class _DetectorProtocol(Protocol):
    def detect(self, filename: str):
        """Detect the channel or source type from the provided filename."""


@dataclass(slots=True)
class UploadDependencies:
    """Dependencies required to process uploaded files."""

    loader: _LoaderProtocol
    detector: _DetectorProtocol


def load_uploaded_files(files: list[UploadFile], dependencies: UploadDependencies) -> list:
    """Persist uploaded files to disk temporarily and return loaded file objects."""

    loaded_files: list = []
    for upload in files:
        temp_path: Path | None = None
        try:
            temp_path = _write_upload_to_tempfile(upload)
            loaded_file = dependencies.loader.load(str(temp_path))
            _apply_filename_detection(upload.filename or "", loaded_file, dependencies.detector)
            loaded_files.append(loaded_file)
        except Exception as exc:
            raise UploadProcessingError(
                f"Failed to load {upload.filename or 'uploaded file'}: {exc}"
            ) from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    return loaded_files


def _write_upload_to_tempfile(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read() if hasattr(upload.file, "read") else b"")
        return Path(tmp.name)


def _apply_filename_detection(filename: str, loaded_file, detector: _DetectorProtocol) -> None:
    channel = getattr(loaded_file, "channel", None)
    unknown_value = getattr(channel, "value", None)
    if unknown_value == "unknown":
        loaded_file.channel = detector.detect(filename)
