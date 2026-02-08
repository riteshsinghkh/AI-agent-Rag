"""
Shipment data extraction helpers.
"""

import re
from typing import Dict, Optional


CURRENCY_CODES = ["USD", "EUR", "GBP", "INR", "CAD", "AUD"]


def _first_match(text: str, patterns) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            return value if value else None
    return None


def _extract_rate_and_currency(text: str) -> (Optional[str], Optional[str]):
    rate_line = _first_match(
        text,
        [
            r"^\s*rate\s*[:#-]\s*(.+)$",
            r"^\s*freight\s*rate\s*[:#-]\s*(.+)$",
        ],
    )

    if not rate_line:
        return None, None

    currency = None
    for code in CURRENCY_CODES:
        if re.search(rf"\b{code}\b", rate_line, flags=re.IGNORECASE):
            currency = code
            break

    if currency is None:
        if "$" in rate_line:
            currency = "USD"
        elif "eur" in rate_line.lower():
            currency = "EUR"
        elif "gbp" in rate_line.lower():
            currency = "GBP"

    rate_match = re.search(r"([0-9]+(?:[\.,][0-9]+)?)", rate_line)
    rate_value = rate_match.group(1) if rate_match else rate_line

    return rate_value.strip() if rate_value else None, currency


def extract_shipment_fields(text: str) -> Dict[str, Optional[str]]:
    """
    Extract structured shipment fields from raw text.

    Returns a dict with nulls for missing fields.
    """
    shipment_id = _first_match(
        text,
        [
            r"^\s*shipment\s*(?:id|no\.?|number)\s*[:#-]\s*(.+)$",
            r"^\s*reference\s*(?:id|no\.?|number)?\s*[:#-]\s*(.+)$",
        ],
    )

    shipper = _first_match(
        text,
        [
            r"^\s*shipper\s*[:#-]\s*(.+)$",
            r"^\s*shipper\s*name\s*[:#-]\s*(.+)$",
        ],
    )

    consignee = _first_match(
        text,
        [
            r"^\s*consignee\s*[:#-]\s*(.+)$",
            r"^\s*consignee\s*name\s*[:#-]\s*(.+)$",
        ],
    )

    pickup_datetime = _first_match(
        text,
        [
            r"^\s*pickup\s*(?:date|datetime|time)\s*[:#-]\s*(.+)$",
            r"^\s*pickup\s*[:#-]\s*(.+)$",
        ],
    )

    delivery_datetime = _first_match(
        text,
        [
            r"^\s*delivery\s*(?:date|datetime|time)\s*[:#-]\s*(.+)$",
            r"^\s*delivery\s*[:#-]\s*(.+)$",
        ],
    )

    equipment_type = _first_match(
        text,
        [
            r"^\s*equipment\s*(?:type)?\s*[:#-]\s*(.+)$",
            r"^\s*trailer\s*(?:type)?\s*[:#-]\s*(.+)$",
        ],
    )

    mode = _first_match(
        text,
        [
            r"^\s*mode\s*[:#-]\s*(.+)$",
            r"^\s*service\s*level\s*[:#-]\s*(.+)$",
        ],
    )

    weight = _first_match(
        text,
        [
            r"^\s*weight\s*[:#-]\s*(.+)$",
            r"^\s*total\s*weight\s*[:#-]\s*(.+)$",
        ],
    )

    carrier_name = _first_match(
        text,
        [
            r"^\s*carrier\s*(?:name)?\s*[:#-]\s*(.+)$",
            r"^\s*carrier\s*[:#-]\s*(.+)$",
        ],
    )

    rate, currency = _extract_rate_and_currency(text)

    return {
        "shipment_id": shipment_id,
        "shipper": shipper,
        "consignee": consignee,
        "pickup_datetime": pickup_datetime,
        "delivery_datetime": delivery_datetime,
        "equipment_type": equipment_type,
        "mode": mode,
        "rate": rate,
        "currency": currency,
        "weight": weight,
        "carrier_name": carrier_name,
    }
