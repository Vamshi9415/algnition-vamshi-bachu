"""Upload endpoint: accepts CSV files, returns canonical dataset + validation report."""
import tempfile
import os
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from src.ingestion.loader import CSVLoader
from src.canonical.schema import CanonicalSchemaBuilder
from src.validation.validator import DataValidator
from src.preprocessing.cleaner import DataCleaner

router = APIRouter(tags=["upload"])
loader = CSVLoader()
builder = CanonicalSchemaBuilder()
validator = DataValidator()
cleaner = DataCleaner()

_canonical_cache: dict = {}


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    loaded = []
    for upload in files:
        try:
            suffix = Path(upload.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp_path = tmp.name
            lf = loader.load(tmp_path)
            # rename channel based on original filename for detection
            from src.ingestion.detector import SourceDetector
            detector = SourceDetector()
            lf.channel = detector.detect(upload.filename) if lf.channel.value == "unknown" else lf.channel
            loaded.append(lf)
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load {upload.filename}: {e}")

    try:
        canonical = builder.build_from_many(loaded)
        canonical = cleaner.clean(canonical)
        report = validator.validate(canonical)
        _canonical_cache["df"] = canonical
        return JSONResponse({
            "status": "success",
            "rows": len(canonical),
            "channels": canonical["channel"].unique().tolist(),
            "campaigns": int(canonical["campaign_name"].nunique()),
            "date_range": {
                "min": str(canonical["date"].min().date()),
                "max": str(canonical["date"].max().date()),
            },
            "validation": report.to_dict(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
