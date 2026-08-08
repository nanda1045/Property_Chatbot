from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.auth import local_authenticated_user
from app.core.authorization import AuthorizationContext
from app.core.config import Settings
from app.core.logging import JsonLogFormatter, request_id_context
from app.core.rate_limit import RateLimitDecision
from app.main import _resolve_agent_approval, app
from app.schemas import ChatResponse


class ProductionHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)
        self.previous_limiter = app.state.rate_limiter

    def tearDown(self) -> None:
        app.state.rate_limiter = self.previous_limiter

    def test_settings_normalize_values_and_reject_invalid_environment(self) -> None:
        settings = Settings(
            _env_file=None,
            app_env="PRODUCTION",
            app_reload=False,
            log_level="warning",
            mysql_password="not-logged",
            auth_mode="entra",
            entra_tenant_id="00000000-0000-0000-0000-000000000001",
            entra_api_audience="00000000-0000-0000-0000-000000000002",
        )

        self.assertEqual(settings.app_env, "production")
        self.assertEqual(settings.log_level, "WARNING")
        self.assertNotIn("not-logged", repr(settings))
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, mysql_port=0)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, app_env="qa")
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, app_env="production", app_reload=True)
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, local_auth_user_id="   ")

    def test_liveness_does_not_require_database_and_readiness_does(self) -> None:
        health = self.client.get("/health")
        with patch("app.main.MySQLDatabase.ping", return_value=True):
            ready = self.client.get("/ready")
        with patch("app.main.MySQLDatabase.ping", side_effect=OSError("offline")):
            unavailable = self.client.get("/ready")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["checks"], {"database": "ok"})
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["error"]["code"], "service_unavailable")

    def test_request_ids_and_validation_errors_use_stable_envelope(self) -> None:
        response = self.client.post(
            "/chat",
            headers={"X-Request-ID": "trace-123"},
            json={"property_code": "not valid", "message": ""},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers["x-request-id"], "trace-123")
        self.assertEqual(response.json()["request_id"], "trace-123")
        self.assertEqual(response.json()["error"]["code"], "validation_error")
        self.assertGreaterEqual(len(response.json()["violations"]), 2)

    def test_unexpected_errors_do_not_leak_internal_details(self) -> None:
        with patch(
            "app.main.RentRollRepository.list_properties",
            side_effect=RuntimeError("private database details"),
        ):
            response = self.client.get(
                "/properties",
                headers={"X-Request-ID": "trace-500"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["x-request-id"], "trace-500")
        self.assertEqual(response.json()["error"]["code"], "internal_error")
        self.assertNotIn("private database details", response.text)

    def test_replaceable_rate_limit_hook_returns_retry_contract(self) -> None:
        calls: list[tuple[str, str]] = []

        class DenyLimiter:
            def check(self, *, client_id: str, route: str) -> RateLimitDecision:
                calls.append((client_id, route))
                return RateLimitDecision(allowed=False, retry_after_seconds=9)

        app.state.rate_limiter = DenyLimiter()
        response = self.client.post(
            "/chat",
            json={"property_code": "115r", "message": "Show occupancy"},
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "9")
        self.assertEqual(response.json()["error"]["code"], "rate_limited")
        self.assertEqual(calls[0][1], "/chat")

    def test_approval_decision_emits_sanitized_audit_log(self) -> None:
        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def resolve_sql_approval(self, **kwargs) -> ChatResponse:
                return ChatResponse(
                    property_code="115r",
                    model="mock:test",
                    conversation_id="conversation-1",
                    run_id="run-1",
                    run_status="completed",
                    answer_markdown="Completed.",
                )

        with (
            patch("app.main.AgentRuntime", FakeRuntime),
            self.assertLogs("aker.audit", level="INFO") as captured,
        ):
            settings = Settings(_env_file=None)
            user = local_authenticated_user(settings)
            _resolve_agent_approval(
                run_id="run-1",
                property_code="115r",
                conversation_id="conversation-1",
                approved=True,
                settings=settings,
                authorization_context=AuthorizationContext.from_settings(
                    user,
                    settings,
                    property_code="115r",
                ),
            )

        self.assertIn("sql_approval_decision", captured.output[0])
        self.assertNotIn("SELECT", captured.output[0])

    def test_json_log_formatter_includes_request_context(self) -> None:
        record = logging.LogRecord(
            "aker.test",
            logging.INFO,
            __file__,
            1,
            "completed",
            (),
            None,
        )
        record.event = "test_completed"
        token = request_id_context.set("trace-log")
        try:
            payload = json.loads(JsonLogFormatter().format(record))
        finally:
            request_id_context.reset(token)

        self.assertEqual(payload["event"], "test_completed")
        self.assertEqual(payload["request_id"], "trace-log")
        self.assertEqual(payload["level"], "INFO")


if __name__ == "__main__":
    unittest.main()
