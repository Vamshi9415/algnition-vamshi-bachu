"""Upload endpoint: accepts CSV files, returns canonical dataset + validation report."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.api.services.upload_service import (
    UploadDependencies,
    UploadProcessingError,
    load_uploaded_files,
)
from ml_engine.canonical.schema import CanonicalSchemaBuilder
from ml_engine.ingestion.detector import SourceDetector
from ml_engine.ingestion.loader import CSVLoader
from ml_engine.preprocessing.cleaner import DataCleaner
from ml_engine.validation.validator import DataValidator

router = APIRouter(tags=["Upload"])
loader = CSVLoader()
builder = CanonicalSchemaBuilder()
validator = DataValidator()
cleaner = DataCleaner()
detector = SourceDetector()
upload_dependencies = UploadDependencies(loader=loader, detector=detector)

_canonical_cache: dict[str, object] = {}


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        loaded = load_uploaded_files(files, upload_dependencies)
    except UploadProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        canonical = builder.build_from_many(loaded)
        canonical = cleaner.clean(canonical)
        report = validator.validate(canonical)
        _canonical_cache.clear()
        _canonical_cache["df"] = canonical
        return JSONResponse(
            {
                "status": "success",
                "rows": len(canonical),
                "channels": canonical["channel"].unique().tolist(),
                "campaigns": int(canonical["campaign_name"].nunique()),
                "date_range": {
                    "min": str(canonical["date"].min().date()),
                    "max": str(canonical["date"].max().date()),
                },
                "validation": report.to_dict(),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc