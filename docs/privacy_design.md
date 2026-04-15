# CogniTeam — Privacy Design Document

## Core Privacy Principle

> **CogniTeam never stores raw message content.**
> Only communication metadata is persisted. Analysis happens at the aggregate and behavioral level, not the individual message level.

---

## What Is Stored vs. Not Stored

| Data Category | Stored? | Details |
|---------------|---------|---------|
| Message content | ❌ Never | Email bodies and chat messages are never written to any database |
| Message metadata | ✅ Yes | Word count, timestamp, sender/receiver IDs, response time |
| Employee names | 🔒 Hashed | SHA-256 hash → anonymous ID (e.g., `Emp_a4f2b3c9`) |
| Email addresses | ❌ Stripped | Removed at ingestion layer by `anonymizer.py` |
| Sentiment scores | ✅ Aggregated | Weekly aggregate only, never per-message |
| Behavioral features | ✅ Yes | Statistical summaries of communication patterns |
| Health/risk scores | ✅ + DP noise | Individual scores stored with differential privacy noise applied at API layer |
| SHAP explanations | ✅ JSONB | Feature importance values, no raw message reference |
| Audit logs | ✅ Yes | Who accessed what, when, from which IP |

---

## Differential Privacy Implementation

CogniTeam uses the `diffprivlib` library to protect individual privacy in all exposed scores.

**Implementation:**
- **Mechanism:** Laplace noise mechanism
- **Privacy budget (ε):** 0.1 (very strong protection — see below)
- **Sensitivity:** 1.0 (max score change from one person's data)
- **Applied:** At the API response layer before any data is returned to the client

**Epsilon scale interpretation:**
| ε value | Privacy level | Noise magnitude |
|---------|---------------|-----------------|
| 0.01 | Very strong | ±10 on a 0–1 scale |
| 0.1 | Strong | ±1 on a 0–1 scale |
| 1.0 | Moderate | ±0.1 on a 0–1 scale |
| 10.0 | Weak | ±0.01 on a 0–1 scale |

CogniTeam uses ε=0.1, making individual scores mathematically indistinguishable
from their true value within ±1 unit of the sensitivity range.

**Code location:** `src/privacy/differential_privacy.py`

---

## Role-Based Access Control

Three user roles enforce data minimization:

| Feature | Employee | Manager | HR |
|---------|----------|---------|-----|
| Own health scores | View (noisy) | — | — |
| All employees' data | ❌ | Team only | ✅ Full |
| SHAP explanations | ❌ | Flags only | ✅ Full |
| Audit trail | Own only | Team only | ✅ Full |
| Raw behavioral features | ❌ | Aggregated | ✅ Full |
| Chat AI queries | ❌ | Team context | ✅ All context |

**Code location:** `src/api/middleware/auth.py`, `src/agents/privacy_agent.py`

---

## GDPR Compliance Decisions

### 1. Right to Access (Article 15)
- Every data access is logged in `audit_log` table
- `audit_logger.get_access_summary(employee_id)` returns who accessed an employee's data
- Export function generates CSV report of all accesses

### 2. Data Minimization (Article 5c)
- Only metadata stored, never message content
- Aggregated weekly features rather than per-message features
- Employee names consistently hashed — irreversible without the salt

### 3. Purpose Limitation (Article 5b)
- Data collected solely for organizational health monitoring
- Not shared with third parties
- Not used for individual performance reviews without HR approval

### 4. Storage Limitation (Article 5e)
- Historical data can be deleted per employee via the audit API
- Retention policy: 2 years recommended (configure in PostgreSQL)

### 5. Right to Erasure (Article 17)
- `DELETE FROM employees WHERE id = :id CASCADE` removes all associated records
- ChromaDB vectors for the employee are removed from all collections

### 6. Transparency
- Every API response includes `privacy_reviewed: true` flag
- AI responses always include an explicit caveat about AI limitations
- Dashboard shows data collection notice on first login

---

## Audit Logging Design

Every database read and API call is logged with:
- `user_id`: who performed the access
- `action`: what was done (e.g., `READ:health_scores`, `API:/employees/42`)
- `target`: which record was accessed
- `timestamp`: UTC timestamp
- `ip_address`: client IP

Logs are never deleted. They are separate from operational data and are
append-only to support forensic analysis.

**Code location:** `src/privacy/audit_logger.py`, `src/api/middleware/privacy.py`

---

## Name Anonymization

Employee names are hashed using SHA-256 with an application-specific salt:

```
anonymized = SHA-256(salt + ":" + name.lower()) → first 10 hex chars
display = "Emp_" + anonymized
```

Properties:
- **Deterministic:** Same name always maps to same anonymous ID
- **Non-reversible:** Cannot reconstruct the name from the hash without the salt
- **Consistent:** The same employee is always shown as the same ID across all views
- **Salt protection:** Even with access to the hash function, the salt prevents rainbow table attacks

The salt is stored only in the application environment (`.env`) and never in the database.

---

## Security Considerations

1. **JWT tokens** expire after 60 minutes (configurable)
2. **Passwords** are hashed with bcrypt (cost factor 12)
3. **All connections** should use HTTPS in production (configure at nginx/load balancer layer)
4. **Database** credentials should rotate regularly
5. **Ollama** runs locally — no data ever sent to external LLM APIs
6. **ChromaDB** vector embeddings do not contain raw text — only semantic vectors

---

## Contact

For privacy inquiries or data subject access requests, contact the platform administrator.
