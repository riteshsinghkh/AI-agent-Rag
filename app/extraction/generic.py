"""
Generic text extraction helpers for any document.
"""

from typing import List, Dict


def extract_key_values(text: str, max_items: int = 20) -> List[Dict[str, str]]:
    """Extract simple key: value pairs from text lines."""
    results: List[Dict[str, str]] = []
    seen = set()

    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if len(key) > 50 or len(value) > 200:
            continue
        signature = f"{key}:{value}"
        if signature in seen:
            continue
        seen.add(signature)
        results.append({"key": key, "value": value})
        if len(results) >= max_items:
            break

    return results


def build_text_preview(text: str, max_chars: int = 1200) -> str:
    """Return a short preview of text."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
