"""Structured output contract for invoice extraction (Engine B)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "total_amount", "currency"]


class InvoiceFields(BaseModel):
    """Gemini must answer with exactly this JSON object (response_schema)."""

    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
