"""
JWT Authentication Middleware
==============================
Access / refresh token pair with server-side rotation and revocation.

Token design:
  access_token  — short-lived JWT (30 min), carried in Authorization header
  refresh_token — long-lived opaque token (7 days), stored as httpOnly cookie
                  SHA-256 hash stored in refresh_tokens table (raw never persisted)

Security properties:
  • Token rotation  — every /auth/refresh issues a NEW refresh token and revokes the old one
  • Revocation      — tokens invalidated server-side in DB (logout, suspicious reuse)
  • Reuse detection — if a revoked token is presented, ALL tokens for that user are revoked
  • Clean-up        — expired/revoked tokens are pruned on login / refresh
"""
from __future__ import annotations

import hashlib
import os
import secrets
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM    = "HS256"

# Access token: 30 minutes (was 60 — short window reduces blast radius)
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Refresh token: 7 days
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cogniteam:cogniteam@localhost:5432/cogniteam")

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
bearer_scheme = HTTPBearer()


# ─────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────

class TokenData(BaseModel):
    user_id:     int
    email:       str
    role:        str
    employee_id: Optional[int] = None


class UserInDB(BaseModel):
    id:              int
    email:           str
    hashed_password: str
    role:            str
    employee_id:     Optional[int]
    is_active:       bool


# ─────────────────────────────────────────────────────────────
# DB schema bootstrap  (idempotent — safe to run on every start)
# ─────────────────────────────────────────────────────────────

def _ensure_db_schema() -> None:
    """Create the refresh_tokens table if it doesn't yet exist."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash  VARCHAR(64) NOT NULL UNIQUE,
                expires_at  TIMESTAMP WITH TIME ZONE NOT NULL,
                revoked     BOOLEAN DEFAULT FALSE,
                created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rt_user   ON refresh_tokens(user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rt_hash   ON refresh_tokens(token_hash)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rt_expiry ON refresh_tokens(expires_at)"))
        conn.commit()


# ─────────────────────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─────────────────────────────────────────────────────────────
# Access token  (JWT)
# ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire    = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"]  = expire
    to_encode["type"] = "access"
    # jti makes every token unique, preventing equality even within the same second
    to_encode["jti"]  = str(_uuid.uuid4())
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise exc
        user_id: Optional[int] = payload.get("sub")
        email:   Optional[str] = payload.get("email")
        role:    Optional[str] = payload.get("role")
        if user_id is None or role is None:
            raise exc
        return TokenData(
            user_id=int(user_id),
            email=email or "",
            role=role,
            employee_id=payload.get("employee_id"),
        )
    except JWTError:
        raise exc


# ─────────────────────────────────────────────────────────────
# Refresh token  (opaque, DB-backed)
# ─────────────────────────────────────────────────────────────

def _hash_token(raw: str) -> str:
    """SHA-256 hex digest — only the hash is ever stored."""
    return hashlib.sha256(raw.encode()).hexdigest()


def create_refresh_token() -> str:
    """Generate a cryptographically secure random token (512 bits of entropy)."""
    return secrets.token_urlsafe(64)


def store_refresh_token(user_id: int, raw_token: str) -> datetime:
    """
    Persist a hashed refresh token for `user_id`.

    Returns the token's expiry datetime so the caller can set cookie max_age.
    Also prunes expired/revoked tokens for this user (lazy cleanup).
    """
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        # Lazy cleanup: only remove EXPIRED tokens.
        # Revoked-but-not-yet-expired tokens must stay in the DB so that
        # the reuse-detection logic in verify_and_rotate_refresh_token can
        # fire the cascade if someone presents a stolen revoked token.
        session.execute(
            text("""
                DELETE FROM refresh_tokens
                WHERE user_id = :uid
                  AND expires_at < NOW()
            """),
            {"uid": user_id},
        )
        session.execute(
            text("""
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (:uid, :hash, :exp)
            """),
            {"uid": user_id, "hash": _hash_token(raw_token), "exp": expires_at},
        )
        session.commit()
    return expires_at


def verify_and_rotate_refresh_token(raw_token: str) -> Optional[UserInDB]:
    """
    Validate a refresh token and perform rotation:
      1. Look up the token hash in the DB.
      2. Check it is not revoked and not expired.
      3. Mark it revoked (so it can never be reused).
      4. Return the associated user so the caller can issue a fresh pair.

    Security — reuse detection:
      If a revoked token is presented, all tokens for that user are immediately
      revoked, forcing re-authentication. This limits damage from token theft.

    Returns None on any failure (expired, revoked, not found).
    """
    token_hash = _hash_token(raw_token)
    engine     = create_engine(DATABASE_URL)

    with Session(engine) as session:
        row = session.execute(
            text("""
                SELECT id, user_id, expires_at, revoked
                FROM refresh_tokens
                WHERE token_hash = :hash
            """),
            {"hash": token_hash},
        ).fetchone()

        if not row:
            return None

        rt_id, user_id, expires_at, revoked = row

        # Reuse of a revoked token → nuke every session for this user
        if revoked:
            session.execute(
                text("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = :uid"),
                {"uid": user_id},
            )
            session.commit()
            return None

        # Expired
        if expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            session.execute(
                text("UPDATE refresh_tokens SET revoked = TRUE WHERE id = :id"),
                {"id": rt_id},
            )
            session.commit()
            return None

        # Rotate: revoke the just-used token
        session.execute(
            text("UPDATE refresh_tokens SET revoked = TRUE WHERE id = :id"),
            {"id": rt_id},
        )
        session.commit()

    return _get_user_by_id(user_id)


def revoke_refresh_token(raw_token: str) -> None:
    """Revoke a specific refresh token (e.g. on logout from one device)."""
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        session.execute(
            text("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = :hash"),
            {"hash": _hash_token(raw_token)},
        )
        session.commit()


def revoke_all_user_tokens(user_id: int) -> None:
    """Revoke every active session for a user (e.g. 'log out of all devices')."""
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        session.execute(
            text("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = :uid"),
            {"uid": user_id},
        )
        session.commit()


# ─────────────────────────────────────────────────────────────
# User lookup helpers
# ─────────────────────────────────────────────────────────────

def _get_user_by_id(user_id: int) -> Optional[UserInDB]:
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        row = session.execute(
            text("""
                SELECT id, email, hashed_password, role, employee_id, is_active
                FROM users WHERE id = :id
            """),
            {"id": user_id},
        ).fetchone()
    if not row:
        return None
    return UserInDB(
        id=row[0], email=row[1], hashed_password=row[2],
        role=row[3], employee_id=row[4], is_active=row[5],
    )


def get_user_by_email(email: str) -> Optional[UserInDB]:
    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        row = session.execute(
            text("""
                SELECT id, email, hashed_password, role, employee_id, is_active
                FROM users WHERE email = :email
            """),
            {"email": email},
        ).fetchone()
    if not row:
        return None
    return UserInDB(
        id=row[0], email=row[1], hashed_password=row[2],
        role=row[3], employee_id=row[4], is_active=row[5],
    )


def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# ─────────────────────────────────────────────────────────────
# FastAPI dependencies
# ─────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> TokenData:
    return decode_token(credentials.credentials)


def require_role(*allowed_roles: str):
    """
    Returns a FastAPI dependency that enforces role-based access.

    Usage:
        @router.get("/hr-only")
        async def hr_only(user: TokenData = Depends(require_role("hr"))):
            ...
    """
    async def role_check(user: TokenData = Depends(get_current_user)) -> TokenData:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied. "
                    f"Required roles: {list(allowed_roles)}. Your role: {user.role}"
                ),
            )
        return user

    return role_check
