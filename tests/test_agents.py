"""
Tests for the multi-agent LLM coordinator pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch


SAMPLE_EMPLOYEE_DATA = {
    "employee_id": 42,
    "profile": {"name": "Emp_abc123", "department": "Engineering", "role": "Developer", "tenure_months": 18},
    "health_scores": {
        "burnout_risk": 0.78,
        "attrition_risk_30d": 0.45,
        "attrition_risk_60d": 0.55,
        "attrition_risk_90d": 0.65,
        "conflict_risk": 0.30,
        "engagement_score": 0.35,
        "overall_score": 38,
        "flags": ["high_after_hours", "declining_sentiment"],
        "shap_values": {"satisfaction_level": -0.4, "average_montly_hours": 0.3},
        "week": "2024-01-15",
    },
    "behavioral_features": {
        "avg_response_hours": 18.5,
        "after_hours_count": 22,
        "message_count": 45,
        "avg_message_length": 15.2,
        "participation_rate": 0.4,
        "manager_comm_ratio": 0.1,
        "sentiment_velocity": -0.31,
        "week": "2024-01-15",
    },
    "active_alerts": [
        {"type": "burnout", "severity": "critical", "description": "High burnout risk detected."}
    ],
}

SAMPLE_ANALYST_OUTPUT = {
    "question": "Is employee 42 at risk of burnout?",
    "employee_summary": SAMPLE_EMPLOYEE_DATA,
    "behavioral_trends": [],
}

SAMPLE_PSYCH_OUTPUT = {
    "interpretation": "The employee shows multiple high-risk behavioral indicators including extended after-hours activity and declining sentiment.",
    "risk_level": "critical",
    "data_summary": "Burnout risk: 78%, After-hours: 22 messages, Sentiment declining.",
}

SAMPLE_ADVISOR_OUTPUT = {
    "recommendations": [
        {"what": "Schedule immediate 1-on-1", "why": "Critical risk", "timeline": "24h", "script": "I wanted to check in."},
        {"what": "Redistribute workload", "why": "Overload detected", "timeline": "This week", "script": "Let's move some tasks."},
        {"what": "Refer to EAP", "why": "Mental health support", "timeline": "48h", "script": "We have resources available."},
    ],
    "summary": "3 urgent actions recommended.",
}


class TestCoordinator:
    def test_all_agents_called(self):
        """All 4 agents (analyst, psychologist, advisor, privacy) must be invoked."""
        with (
            patch("src.agents.coordinator.analyst_agent") as mock_analyst,
            patch("src.agents.coordinator.psychologist_agent") as mock_psych,
            patch("src.agents.coordinator.advisor_agent") as mock_advisor,
            patch("src.agents.coordinator.privacy_agent") as mock_privacy,
            patch("src.agents.coordinator.Ollama") as mock_llm,
        ):
            mock_llm.return_value.invoke.return_value = "employee_id: 42\nteam_id: null\nintent: burnout"
            mock_analyst.run.return_value = SAMPLE_ANALYST_OUTPUT
            mock_psych.run.return_value = SAMPLE_PSYCH_OUTPUT
            mock_advisor.run.return_value = SAMPLE_ADVISOR_OUTPUT
            mock_privacy.run.return_value = {
                "answer": "The employee shows burnout signals.",
                "recommendations": SAMPLE_ADVISOR_OUTPUT["recommendations"],
                "risk_level": "critical",
                "privacy_reviewed": True,
            }

            from src.agents.coordinator import answer
            result = answer("Is employee 42 at risk of burnout?", user_role="manager")

        mock_analyst.run.assert_called_once()
        mock_psych.run.assert_called_once()
        mock_advisor.run.assert_called_once()
        mock_privacy.run.assert_called_once()

    def test_response_contains_recommendations(self):
        """Final response must contain a recommendations list."""
        with (
            patch("src.agents.coordinator.analyst_agent") as mock_analyst,
            patch("src.agents.coordinator.psychologist_agent") as mock_psych,
            patch("src.agents.coordinator.advisor_agent") as mock_advisor,
            patch("src.agents.coordinator.privacy_agent") as mock_privacy,
            patch("src.agents.coordinator.Ollama") as mock_llm,
        ):
            mock_llm.return_value.invoke.return_value = "employee_id: null\nteam_id: null\nintent: general"
            mock_analyst.run.return_value = SAMPLE_ANALYST_OUTPUT
            mock_psych.run.return_value = SAMPLE_PSYCH_OUTPUT
            mock_advisor.run.return_value = SAMPLE_ADVISOR_OUTPUT
            mock_privacy.run.return_value = {
                "answer": "Some answer",
                "recommendations": SAMPLE_ADVISOR_OUTPUT["recommendations"],
                "risk_level": "high",
                "privacy_reviewed": True,
            }

            from src.agents.coordinator import answer
            result = answer("What should I do about my team?", user_role="manager")

        assert "recommendations" in result, "Response must contain 'recommendations'"
        assert len(result["recommendations"]) > 0, "Recommendations list must not be empty"

    def test_privacy_agent_removes_pii(self):
        """Privacy agent should not return raw email addresses."""
        from src.agents.privacy_agent import run as privacy_run

        contaminated = {
            "answer": "The employee john.doe@company.com is at risk.",
            "recommendations": [],
        }
        cleaned = privacy_run(contaminated, user_role="manager")
        assert "john.doe@company.com" not in cleaned.get("answer", ""), \
            "Email address should be redacted in privacy-reviewed response"
        assert "[REDACTED EMAIL]" in cleaned.get("answer", "")

    def test_privacy_reviewed_flag_set(self):
        """privacy_reviewed flag must be True in final response."""
        from src.agents.privacy_agent import run as privacy_run

        response = {"answer": "All looks good.", "recommendations": []}
        result = privacy_run(response, user_role="manager")
        assert result.get("privacy_reviewed") is True

    def test_ai_caveat_added_to_answer(self):
        """AI caveat should be appended to answer text."""
        from src.agents.privacy_agent import run as privacy_run

        response = {"answer": "Employee shows burnout risk.", "recommendations": []}
        result = privacy_run(response, user_role="hr")
        assert "AI Note" in result.get("answer", "") or "⚠️" in result.get("answer", ""), \
            "AI caveat should be present in answer"
