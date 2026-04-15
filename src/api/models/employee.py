"""Pydantic models for employee-related API requests and responses."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class EmployeeBase(BaseModel):
    name: str
    department: Optional[str] = None
    role: Optional[str] = None
    manager_id: Optional[int] = None
    tenure_months: Optional[int] = 0


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeResponse(EmployeeBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class HealthScoreResponse(BaseModel):
    employee_id: int
    week: Optional[date] = None
    burnout_risk: Optional[float] = None
    attrition_risk_30d: Optional[float] = None
    attrition_risk_60d: Optional[float] = None
    attrition_risk_90d: Optional[float] = None
    conflict_risk: Optional[float] = None
    engagement_score: Optional[float] = None
    overall_score: Optional[int] = None
    shap_values: Optional[dict] = None
    flags: Optional[list] = None


class BehavioralFeatureResponse(BaseModel):
    employee_id: int
    week: Optional[date] = None
    avg_response_hours: Optional[float] = None
    after_hours_count: Optional[int] = None
    message_count: Optional[int] = None
    avg_message_length: Optional[float] = None
    participation_rate: Optional[float] = None
    manager_comm_ratio: Optional[float] = None
    sentiment_velocity: Optional[float] = None


class EmployeeDetailResponse(BaseModel):
    employee: EmployeeResponse
    health_scores: Optional[HealthScoreResponse] = None
    behavioral_features: Optional[BehavioralFeatureResponse] = None
    sentiment_trend: list[dict] = []
    active_alerts: list[dict] = []
    risk_factors: list[dict] = []


class TeamHealthResponse(BaseModel):
    team_id: int
    avg_health_score: float
    member_count: int
    members: list[dict] = []
    top_alerts: list[dict] = []
    graph_data: Optional[dict] = None
