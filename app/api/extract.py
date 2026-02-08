"""
Extract Endpoint
Handles POST /extract requests for structured extraction.
"""

import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.extraction.generic import extract_key_values, build_text_preview
from app.extraction.shipment import extract_shipment_fields
from app.rag.parsing import parse_document
from app.uploads_state import get_latest_uploads

logger = logging.getLogger(__name__)
router = APIRouter()


class ExtractRequest(BaseModel):
    """Request model for /extract endpoint"""
    text: Optional[str] = Field(
        default=None,
        max_length=20000,
        description="Optional text to extract from"
    )
    use_latest_upload: bool = Field(
        default=True,
        description="Use the latest uploaded file when text is not provided"
    )


class ShipmentExtraction(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[str] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None


class KeyValue(BaseModel):
    key: str
    value: str


class ExtractResponse(BaseModel):
    """Response model for /extract endpoint"""
    text_preview: str
    key_values: List[KeyValue]
    shipment: ShipmentExtraction


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")


@router.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        200: {"description": "Successful response", "model": ExtractResponse},
        400: {"description": "Bad request", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Extract document fields",
    description="Extract generic key-values and shipment fields from text or the latest upload."
)
async def extract_shipment(request: ExtractRequest) -> ExtractResponse:
    """
    Extract structured data from text or latest upload.
    """
    text = (request.text or "").strip()
    if not text and request.use_latest_upload:
        upload_dir = Path(settings.UPLOAD_DIR)
        latest_files = get_latest_uploads(upload_dir)
        if not latest_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No uploaded files available for extraction"
            )

        latest_path = latest_files[-1]
        text = (parse_document(latest_path) or "").strip()

    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text available for extraction"
        )

    if len(text) > 20000:
        text = text[:20000]

    try:
        shipment = extract_shipment_fields(text)
        preview = build_text_preview(text)
        key_values = extract_key_values(text)
        return ExtractResponse(
            text_preview=preview,
            key_values=key_values,
            shipment=ShipmentExtraction(**shipment)
        )
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Extraction failed"
        ) from exc
