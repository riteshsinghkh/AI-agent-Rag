"""
Upload Endpoint
Handles POST /upload requests for adding new documents.
"""

import logging
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.parsing import SUPPORTED_EXTENSIONS
from app.rag.retriever import ingest_documents
from app.uploads_state import record_latest_uploads

logger = logging.getLogger(__name__)
router = APIRouter()


class UploadResponse(BaseModel):
    """Response model for /upload endpoint"""

    stored_files: List[str] = Field(..., description="Stored filenames")
    ingested_files: List[str] = Field(..., description="Files successfully indexed")
    skipped_files: List[str] = Field(..., description="Files skipped (empty or unreadable)")
    invalid_files: List[str] = Field(..., description="Files rejected due to invalid type")
    too_large_files: List[str] = Field(..., description="Files rejected due to size limit")
    chunks_added: int = Field(..., description="Total chunks added to the index")
    index_size: int = Field(..., description="Total vectors in the index after update")


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    invalid_files: List[str] = Field(default=[], description="Invalid file names")
    too_large_files: List[str] = Field(default=[], description="Oversized file names")


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"description": "Bad request", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Upload documents",
    description="Upload TXT, PDF, or DOCX documents and ingest them into the index."
)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload documents, store them, and trigger ingestion.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_files: List[str] = []
    saved_paths: List[Path] = []
    invalid_files: List[str] = []
    too_large_files: List[str] = []
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    for upload in files:
        if not upload.filename:
            invalid_files.append("unknown")
            continue

        filename = Path(upload.filename).name
        suffix = Path(filename).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            invalid_files.append(filename)
            continue

        unique_name = f"{Path(filename).stem}-{uuid4().hex}{suffix}"
        dest_path = upload_dir / unique_name

        try:
            content = await upload.read()

            if len(content) > max_bytes:
                too_large_files.append(filename)
                continue

            dest_path.write_bytes(content)
        except Exception as exc:
            logger.error(f"Failed to store upload {filename}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store file: {filename}"
            ) from exc

        stored_files.append(unique_name)
        saved_paths.append(dest_path)

    if not saved_paths:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": "No valid files to process",
                "invalid_files": invalid_files,
                "too_large_files": too_large_files
            }
        )

    try:
        ingest_result = ingest_documents(saved_paths)
    except Exception as exc:
        logger.error(f"Failed to ingest uploaded documents: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest uploaded documents"
        ) from exc

    try:
        record_latest_uploads(stored_files, upload_dir)
    except Exception as exc:
        logger.warning(f"Failed to record latest uploads: {exc}")

    return UploadResponse(
        stored_files=stored_files,
        ingested_files=ingest_result["ingested_files"],
        skipped_files=ingest_result["skipped_files"],
        invalid_files=invalid_files,
        too_large_files=too_large_files,
        chunks_added=ingest_result["chunks_added"],
        index_size=ingest_result["index_size"],
    )
