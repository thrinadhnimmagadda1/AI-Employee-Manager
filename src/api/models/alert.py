"""Pydantic models for alerts API."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AlertResponse(BaseModel):
    id: int
    employee_id: int
    alert_type: str
    severity: str
    description: Optional[str] = None
    recommendations: Optional[list] = None
    resolved: bool = False
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AlertResolveRequest(BaseModel):
    resolved: bool = True
    resolution_note: Optional[str] = None
