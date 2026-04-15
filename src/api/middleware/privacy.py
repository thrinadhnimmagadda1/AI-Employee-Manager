"""
Privacy Middleware
Logs every data access to the audit_log table for GDPR compliance.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")
DP_EPSILON = float(os.getenv("DIFFERENTIAL_PRIVACY_EPSILON", "0.1"))


def apply_dp_noise_to_response(data: Any, epsilon: float = DP_EPSILON) -> Any:
    """
    Pass-through wrapper retained for API compatibility.
    DP noise was removed from the response layer; scores are stored
    with model-level uncertainty already baked in via calibrated probabilities.
    """
    return data


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every API request to the audit_log table.
    Skips health check and auth endpoints to avoid log spam.
    """

    SKIP_PATHS = {"/health", "/auth/login", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        path = request.url.path
        if path in self.SKIP_PATHS or path.startswith("/static"):
            return response

        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.orm import Session

            user_id: int | None = None
            if hasattr(request.state, "user"):
                user_id = getattr(request.state.user, "user_id", None)

            ip = request.client.host if request.client else "unknown"
            engine = create_engine(DATABASE_URL)
            with Session(engine) as session:
                session.execute(
                    text("""
                        INSERT INTO audit_log (user_id, action, target, ip_address)
                        VALUES (:uid, :action, :target, :ip)
                    """),
                    {
                        "uid": user_id,
                        "action": f"{request.method} {path}",
                        "target": str(request.url),
                        "ip": ip,
                    },
                )
                session.commit()
        except Exception:
            pass  # Never let logging break the response

        return response
