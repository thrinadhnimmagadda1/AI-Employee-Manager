<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Transformers-FFD21E)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)

# CogniTeam — AI Organizational Intelligence Platform

> Detect team conflict, predict employee burnout, and give managers early warnings weeks before problems escalate.

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/your-username/cogni-team.git
cd cogni-team
cp .env.example .env          # set DATABASE_URL and SECRET_KEY

# 2. Download all datasets (HR Analytics, GoEmotions, Enron instructions, Reddit)
python scripts/download_data.py

# 3. Train all ML models  (~5 min, XGBoost only; add --include-bert for BERT on GPU)
python scripts/train_all_models.py

# 4. Start all services (Postgres, Redis, backend, frontend)
docker-compose up -d

# 5. Open the platform
open http://localhost:3000
```

**Demo login** (pre-seeded):

| Role | Email | Password |
|------|-------|----------|
| HR Admin | hr@cogniteam.ai | hr123 |
| Manager | manager@cogniteam.ai | manager123 |
| Employee | employee@cogniteam.ai | employee123 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        React Frontend (Port 3000)                │
│   Dashboard · TeamHealth · EmployeeDetail · Alerts · AI Chat     │
└────────────────────────┬─────────────────────────────────────────┘
                         │ REST API (JWT + refresh tokens)
┌────────────────────────▼─────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)                    │
│  /dashboard  /employees  /alerts  /chat  /auth  /admin           │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────────────────┐ │
│  │ Privacy │  │ Audit    │  │    Data-Driven Chat Engine      │ │
│  │  RBAC   │  │ Logging  │  │  DB queries → insight builder   │ │
│  └─────────┘  └──────────┘  └─────────────────────────────────┘ │
└───────────┬──────────────────────┬───────────────────────────────┘
            │                      │
┌───────────▼──────────┐  ┌───────▼──────────────────────────────┐
│    ML Layer          │  │       Data Layer                     │
│  XGBoost Burnout     │  │  PostgreSQL · Redis                  │
│  XGBoost Attrition   │  │  Enron Emails · HR Analytics        │
│  XGBoost Conflict    │  │  GoEmotions · Reddit Posts           │
│  BERT Emotions       │  └──────────────────────────────────────┘
│  NetworkX Graph ML   │
│  SHAP Explainability │
└──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Recharts, D3.js, Tailwind CSS |
| Backend | FastAPI, Uvicorn, SQLAlchemy, Celery |
| Database | PostgreSQL 15, Redis 7, ChromaDB |
| NLP | BERT (fine-tuned), VADER, spaCy |
| ML | XGBoost, scikit-learn, SHAP, Prophet |
| Graph | NetworkX, Isolation Forest |
| Privacy | SHA-256 anonymisation, JWT RBAC, audit log |
| Migrations | Alembic (version-controlled schema) |
| Deploy | Docker, docker-compose |

---

## Datasets

All datasets are free to obtain. The download script handles HR Analytics and GoEmotions automatically.

| Dataset | Source | Size | Used For |
|---------|--------|------|----------|
| Enron Emails | [Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) (free) | ~517K emails | NLP pipeline, communication graph ML |
| HR Analytics | [Kaggle](https://www.kaggle.com/datasets/giripujar/hr-analytics) (free) | 14,999 employees | Burnout and attrition model training |
| GoEmotions | [HuggingFace](https://huggingface.co/datasets/go_emotions) (free) | 58K samples | BERT emotion classifier fine-tuning |
| Reddit Workplace | Reddit API (free) | 10K+ posts | Language augmentation, sentiment corpus |

```bash
python scripts/download_data.py          # downloads HR Analytics + GoEmotions automatically
python scripts/download_data.py --only hr          # HR Analytics only
python scripts/download_data.py --only goemotions  # GoEmotions only
python scripts/download_data.py --skip-reddit      # skip Reddit scraping
```

**Enron dataset** (manual — ~1.7 GB, free):
1. [kaggle.com/datasets/wcukierski/enron-email-dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
2. Download, unzip, rename to `enron_emails.csv`, place at `data/raw/enron_emails.csv`

---

## Model Performance

Models trained on 2026-04-07. Full metrics saved to `models/metrics.json`.

| Model | Algorithm | Dataset | Metric | Score |
|-------|-----------|---------|--------|-------|
| Emotion Classifier | BERT fine-tuned | GoEmotions 58K | Macro F1 | ~62%¹ |
| Burnout Predictor | XGBoost + SHAP | HR Analytics 15K | AUC-ROC | **96.6%** |
| Attrition Model | XGBoost calibrated | HR Analytics 15K | AUC-ROC | **96.8%** |
| Conflict Detector | XGBoost + SHAP | Enron comm graph | F1 | 100%²  |

> ¹ BERT fine-tuning requires GPU (~30 min). Expected score based on published BERT-base GoEmotions benchmark.
> Run `python scripts/train_all_models.py` with `--include-bert` to fine-tune and get the real number.
> Score updates automatically in `models/metrics.json` after training completes.

> ² Conflict model trained on HR Analytics proxy labels
> (`satisfaction_level < 0.30 AND average_monthly_hours > 250`).
> The F1=100% reflects proxy-label quality, not ground-truth conflict annotation.
> Replace with labelled Enron conflict data for production use.

### Retrain all models

```bash
python scripts/train_all_models.py           # XGBoost only (~5 min)
python scripts/train_all_models.py --include-bert   # + BERT fine-tune (~30 min, GPU recommended)
```

Metrics are saved to `models/metrics.json` after every run.

---

## Privacy & Security

CogniTeam is built privacy-first. No raw message content is ever stored or processed.

| Control | Implementation | Details |
|---------|---------------|---------|
| Zero content storage | `src/ingestion/enron_loader.py` | Only metadata is persisted: word count, timestamp, response-time hours — never body text |
| Differential privacy | `src/privacy/differential_privacy.py` | Laplace noise (ε = 0.1) applied to every individual score before API response |
| Role-based access | `src/api/middleware/auth.py` | Three tiers: **employee** (own data only) · **manager** (team data) · **HR** (all data + audit log) |
| GDPR audit trail | `audit_log` table | Every data access logged: user_id, action, target, timestamp, IP |
| Name anonymisation | `src/privacy/anonymizer.py` | Personal names replaced with consistent SHA-256 hash — never reversible |
| Secure authentication | JWT (30 min) + refresh tokens (7 days) | httpOnly cookies, server-side revocation, token rotation, cascade invalidation on reuse detection |

---

## Setup Guide

### Step 1 — Download All Datasets

```bash
python scripts/download_data.py
```

This single command:
- Downloads **HR Analytics** from Kaggle automatically (requires `~/.kaggle/kaggle.json`)
- Downloads **GoEmotions** from HuggingFace automatically (no credentials needed)
- Prints manual download instructions for **Enron** (~1.7 GB, Kaggle)
- Scrapes **Reddit** workplace posts if `REDDIT_CLIENT_ID` is set in `.env`

**Kaggle credentials** (needed for HR Analytics auto-download):
1. Go to [kaggle.com/account](https://www.kaggle.com/account) → API → Create New Token
2. Move the downloaded file: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

Expected output after Step 1:
```
  data/raw/hr_analytics.csv        ✅  14999 rows
  data/raw/go_emotions_train.csv   ✅  43410 rows
  data/raw/go_emotions_val.csv     ✅  5426 rows
  data/raw/enron_emails.csv        ✅  517401 rows   (or ❌ MISSING — manual download)
  data/raw/reddit_posts.csv        ✅  1500 rows     (or ⚠️  skipped — no credentials)
```

---

### Step 2 — Load Enron Data & Run Full Analytics Pipeline

> **This is the most important step.** It transforms CogniTeam from a seeded
> demo into a platform powered by real workplace communication patterns.

**Prerequisites:** Download the Enron email dataset first (see Step 1 — Enron section)
and place it at `data/raw/enron_emails.csv`.

```bash
# Full run — clean, load ~517K emails, run 12 weeks of analytics
python scripts/load_enron_data.py

# Quick smoke-test with 50K emails and 4 weeks
python scripts/load_enron_data.py --limit 50000 --weeks 4

# Re-run only the analytics pipeline (emails already in DB)
python scripts/load_enron_data.py --pipeline-only
```

What the script does end-to-end:

| Step | Module | Output table |
|------|--------|-------------|
| Clean raw CSV | `src/ingestion/data_cleaner.py` | `data/processed/emails_clean.csv` |
| Load email metadata | `src/ingestion/enron_loader.py` | `message_metadata` (~517K rows) |
| Feature engineering × 12 weeks | `src/ml/feature_engineering.py` | `behavioral_features` (1,800 rows) |
| Communication graph × 12 weeks | `src/graph/graph_builder.py` | `comm_graph` |
| Network centrality × 12 weeks | `src/graph/network_analyzer.py` | (in-memory, used for ML step) |
| ML predictions → health scores | burnout / attrition / conflict models | `health_scores` (150 rows) |
| Alert generation | thresholds: burnout >70%, attrition 60d >65% | `alerts` |

**Progress output** (printed every 10,000 emails):
```
  Loaded    10,000 / 517,401 emails  ( 1.9%)  ETA: 8m 12s
  Loaded    20,000 / 517,401 emails  ( 3.9%)  ETA: 7m 58s
  ...
```

**Expected runtime:** 15–30 minutes on a MacBook Pro (M-series), ~60 minutes on Intel.

---

### Step 3 — Start All Services

```bash
docker-compose up -d
```

Or run natively (PostgreSQL + Redis must be installed):
```bash
# Backend (migrations run automatically on startup)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd frontend && npm install && npm run dev
```

### Step 4 — Open the Platform

```bash
open http://localhost:3000
```

Demo login credentials (pre-seeded):
| Role | Email | Password |
|------|-------|----------|
| HR Admin | hr@cogniteam.ai | hr123 |
| Manager | manager@cogniteam.ai | manager123 |
| Employee | employee@cogniteam.ai | employee123 |

---

## Train ML Models

After datasets are downloaded (Step 1), train each model:

```bash
# All XGBoost models + metrics.json  (~5 min)
python scripts/train_all_models.py

# Individual models
python -m src.ml.burnout_predictor
python -m src.ml.attrition_model
python -m src.ml.conflict_detector

# BERT fine-tuning on GoEmotions (GPU recommended, ~30 min on CPU)
python -c "from src.nlp.emotion_classifier import fine_tune_emotion_classifier; fine_tune_emotion_classifier(num_epochs=3)"

# Load Enron email metadata into PostgreSQL (requires enron_emails.csv)
python -m src.ingestion.enron_loader
```

---

## API Reference

| Method | Endpoint | Role | Description |
|--------|----------|------|-------------|
| POST | `/auth/login` | Any | Authenticate, receive access + refresh tokens |
| POST | `/auth/refresh` | Any | Rotate refresh token, get new access token |
| POST | `/auth/logout` | Any | Revoke refresh token server-side |
| GET | `/health` | Any | Service health (DB, Redis, Ollama) |
| GET | `/api/info` | Any | System status + model metrics |
| GET | `/dashboard/overview` | Manager+ | All teams with health scores |
| GET | `/dashboard/team/{id}` | Manager+ | Team detail with graph |
| GET | `/dashboard/trends/{id}` | Manager+ | Employee 12-week trends |
| GET | `/employees` | Manager+ | List all employees |
| GET | `/employees/{id}` | Manager+ | Full employee profile |
| GET | `/employees/{id}/history` | Manager+ | 12-week health score history |
| GET | `/employees/{id}/risk-factors` | Manager+ | SHAP risk factor breakdown |
| GET | `/alerts` | Manager+ | All active alerts |
| PATCH | `/alerts/{id}/resolve` | Manager+ | Resolve an alert |
| GET | `/graph/{week}` | Manager+ | D3 graph JSON for a specific week |
| GET | `/audit-log` | HR only | Paginated audit log with filters |
| POST | `/admin/run-pipeline` | HR only | Trigger weekly scoring pipeline |
| POST | `/admin/train-models` | HR only | Retrain all ML models |
| POST | `/chat` | Manager+ | Data-driven insight query (employee_id or team_id) |

---

## Database Migrations (Alembic)

CogniTeam uses [Alembic](https://alembic.sqlalchemy.org/) for version-controlled,
reproducible schema changes. Migration scripts live in `database/migrations/versions/`.

### Common commands

```bash
# Apply all pending migrations (safe to run on every deploy — idempotent)
alembic upgrade head

# Roll back the most recent migration
alembic downgrade -1

# Roll back to a specific revision
alembic downgrade 0001

# Generate a new migration by diffing ORM models against the live DB
alembic revision --autogenerate -m "add_column_employees_slack_id"

# Show full migration history
alembic history

# Show which revision the connected DB is currently at
alembic current

# Generate plain SQL instead of running it (useful for review / audit)
alembic upgrade head --sql
```

### First-time setup on an existing database

If your database was created by running `database/schema.sql` directly (not through
Alembic), mark it as already at the latest revision without re-running the migration:

```bash
alembic stamp head
```

After that, `alembic current` should print `0001 (head)` and future `alembic upgrade head`
calls will only apply new migrations.

### Adding a new migration (workflow)

1. Edit `src/db/models.py` with your schema change (add/remove column, table, index).
2. Run `alembic revision --autogenerate -m "your_description"` — Alembic diffs the ORM
   models against the live database and writes the migration script automatically.
3. Review the generated file in `database/migrations/versions/` before applying.
4. Run `alembic upgrade head` to apply.

### Docker

Migrations run automatically when the backend container starts:

```yaml
# docker-compose.yml (already configured)
command: sh -c "alembic upgrade head && uvicorn src.api.main:app ..."
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Screenshots

> [Placeholder — add screenshots here]

---

## Demo Video

> [Placeholder — add demo video link here]

---

## License

MIT License — See LICENSE file.
