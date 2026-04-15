"""
Alembic environment script for CogniTeam.

Key decisions:
- DATABASE_URL is read from the environment (or .env file) at runtime so
  credentials are never hard-coded or committed.
- All 10 SQLAlchemy ORM models are imported via src.db.models, populating
  Base.metadata so that `alembic revision --autogenerate` can detect schema
  diffs automatically.
- Both online mode (connected to a live DB) and offline mode (generates SQL
  without connecting) are supported.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Project root on sys.path ─────────────────────────────────────────────────
# Ensures `from src.db.models import Base` resolves correctly when Alembic is
# invoked from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Load .env so DATABASE_URL is available ───────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

# ── Import ORM models so autogenerate can see all tables ────────────────────
from src.db.models import Base  # noqa: E402 — must come after sys.path tweak

# ── Alembic Config object ────────────────────────────────────────────────────
config = context.config

# Wire up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the live DATABASE_URL into the config so engine_from_config picks it up.
# This overrides the blank `sqlalchemy.url` in alembic.ini.
_db_url = os.environ.get("DATABASE_URL")
if not _db_url:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Add it to your .env file or export it before running Alembic.\n"
        "Example: DATABASE_URL=postgresql://cogniteam:cogniteam@localhost:5432/cogniteam"
    )
config.set_main_option("sqlalchemy.url", _db_url)

# Target metadata for autogenerate comparisons
target_metadata = Base.metadata


# ── Offline mode ─────────────────────────────────────────────────────────────
# Generates a SQL script without connecting to the database.
# Usage: alembic upgrade head --sql

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas and render CHECK/UNIQUE constraints
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ──────────────────────────────────────────────────────────────
# Connects to the live database and applies migrations directly.

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # One connection per migration run; avoids pool leaks
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
