from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from backend.api.services.upload_service import UploadDependencies, load_uploaded_files


class _FakeChannel:
    def __init__(self, value: str = "unknown") -> None:
        self.value = value


class _FakeLoadedFile:
    def __init__(self) -> None:
        self.channel = _FakeChannel()


class _FakeLoader:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def load(self, path: str):
        self.paths.append(path)
        return _FakeLoadedFile()


class _FakeDetector:
    def detect(self, filename: str):
        return f"detected:{filename}"


def test_load_uploaded_files_cleans_up_tempfile():
    loader = _FakeLoader()
    detector = _FakeDetector()
    upload = UploadFile(filename="bing.csv", file=BytesIO(b"x,y\n1,2\n"))

    loaded = load_uploaded_files([upload], UploadDependencies(loader=loader, detector=detector))

    assert len(loader.paths) == 1
    assert not Path(loader.paths[0]).exists()
    assert loaded[0].channel == "detected:bing.csv"
