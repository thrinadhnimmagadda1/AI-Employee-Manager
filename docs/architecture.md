# CogniTeam — System Architecture

## Overview

CogniTeam is a 5-layer system that processes raw workplace communication metadata
and transforms it into actionable manager insights through AI prediction engines.

---

## System Layers

### Layer 1: Data Ingestion
**Files:** `src/ingestion/`

Raw email data from the Enron dataset is parsed and **only metadata is stored** —
never message content. The pipeline:
1. Parses sender, receiver, timestamp, word count, and after-hours flag
2. Maps email addresses to anonymized employee IDs
3. Computes thread response times (hours between emails in same thread)
4. Writes to `message_metadata` table only
5. Reddit posts are ingested separately for BERT training augmentation

### Layer 2: NLP Processing
**Files:** `src/nlp/`

A fine-tuned BERT model classifies workplace emotions across 7 categories.
The pipeline per employee per week:
1. Loads all message bodies from that week
2. Classifies dominant emotion using fine-tuned BERT (GoEmotions)
3. Computes continuous sentiment score via VADER (blended with BERT)
4. Aggregates to weekly sentiment score and dominant emotion
5. Detects linguistic signals: hedging, urgency, passive-aggression, disengagement

### Layer 3: Graph ML
**Files:** `src/graph/`

Weekly directed communication graphs capture the social fabric of the organization:
- **Nodes:** employees
- **Edges:** message pairs with frequency, sentiment, and response time attributes
- **Centrality:** degree, betweenness, clustering computed weekly
- **Anomaly detection:** Isolation Forest flags employees whose network position changed abnormally
- **Trend analysis:** detects edges where sentiment or frequency deteriorated ≥50%

### Layer 4: ML Prediction Engines
**Files:** `src/ml/`

Six prediction engines each backed by XGBoost trained on real datasets:

| Engine | Model | Training Data | Output |
|--------|-------|--------------|--------|
| Burnout Risk Detector | XGBoost | HR Analytics | 0.0–1.0 score |
| Attrition Predictor | XGBoost + calibration | HR Analytics | 30/60/90d % |
| Conflict Detector | XGBoost | Synthetic pairs + comm graph | 0.0–1.0 score |
| Team Health Score | Aggregation | All models | 0–100 overall |
| Network Isolation | Isolation Forest | Graph features | 0.0–1.0 score |
| Manager Recommender | LLaMA 3 (LLM) | Historical context (RAG) | Natural language |

Every prediction includes SHAP explainability values identifying the top 3 contributing factors.

### Layer 5: API and Presentation
**Files:** `src/api/`, `frontend/`

FastAPI serves all data to the React dashboard:
- JWT authentication with 3 role tiers
- Differential privacy noise applied before every API response
- Audit logging on every data access
- Celery processes weekly background scoring jobs

---

## Data Flow: Raw Email → Dashboard

```
Enron CSV
    │
    ▼ src/ingestion/enron_loader.py
message_metadata table (metadata only, no content)
    │
    ▼ src/nlp/sentiment_model.py (weekly batch)
sentiment_scores table
    │
    ├── ▼ src/ml/feature_engineering.py
    │   behavioral_features table (14 features)
    │
    ├── ▼ src/graph/graph_builder.py
    │   comm_graph table + NetworkX weekly graph
    │
    ▼ src/ml/burnout_predictor.py + attrition_model.py + conflict_detector.py
health_scores table (SHAP values, flags, all 6 scores)
    │
    ▼ src/api/main.py (Celery alert generation)
alerts table
    │
    ▼ FastAPI → React Frontend
```

---

## Multi-Agent LLM System

When a manager asks a natural language question, the coordinator orchestrates 4 specialized agents:

```
Manager Question
      │
      ▼ coordinator.py (intent extraction via LLaMA 3)
      │
      ├── ▼ analyst_agent.py (SQL queries → structured data)
      │
      ├── ▼ psychologist_agent.py (LLaMA 3 interpretation)
      │
      ├── ▼ advisor_agent.py (LLaMA 3 → 3 recommendations with scripts)
      │
      ▼ privacy_agent.py (PII removal, caveats, role filtering)
      │
      ▼ Final Response to Manager
```

Optionally, RAG retrieves similar historical situations from ChromaDB before generation.

---

## 6 Prediction Engines Detail

### 1. Burnout Risk Detector
- **Training:** HR Analytics (14,999 employees, satisfaction/hours/projects/tenure)
- **Features:** All 14 behavioral features + HR proxy features
- **Output:** Risk score 0.0–1.0 + top-3 SHAP explanations
- **Threshold:** 0.70 = high alert, 0.85 = critical alert

### 2. Attrition / Flight Risk Predictor
- **Training:** HR Analytics `left` label (binary)
- **Calibration:** Sigmoid calibration for reliable probabilities
- **Output:** Three separate horizon predictions (30/60/90 day)
- **Note:** Uses decay factors to map annual risk to each horizon

### 3. Conflict Detector
- **Training:** Communication pair features (response trends, sentiment trends, PA score)
- **Unique:** Operates at the dyadic level (pairs of employees, not individuals)
- **Output:** Conflict probability per pair, flagged if ≥ 0.60

### 4. Team Health Score
- **Computation:** Weighted average of all 6 individual scores for the team
- **Scale:** 0–100 (green ≥70, yellow 40–69, red <40)

### 5. Network Isolation Detector
- **Method:** Graph centrality below threshold + Isolation Forest anomaly detection
- **Signals:** Degree centrality < 0.05, disconnected from team, no response to messages

### 6. Manager Action Recommender
- **Method:** LLaMA 3 (Ollama, fully local) + RAG retrieval from historical outcomes
- **Output:** 3 structured recommendations with what/why/timeline/conversation script
- **Fallback:** Rule-based recommendations when LLM unavailable
