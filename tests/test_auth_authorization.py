from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from app.agents.runtime import AgentRunNotFoundError, AgentRuntime
from app.core.auth import (
    AuthenticatedUser,
    AuthenticationError,
    EntraTokenValidator,
    Role,
    get_authenticated_user,
)
from app.core.authorization import (
    AuthorizationContext,
    AuthorizationDeniedError,
    ToolPermission,
)
from app.core.config import Settings, get_settings
from app.main import app
from app.schemas import ChatResponse
from app.tools.contracts import ToolSpec, TrustedToolContext
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from tests.test_agent_runtime import FakeWorkflow, RecordingRunStore


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AllowedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool


def user_for(role: Role, *, user_id: str = "entra-user") -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=user_id,
        display_name="Security Test User",
        email="security@example.test",
        tenant_id="tenant-1",
        roles=(role,),
    )


def authorization_for(
    role: Role,
    *,
    user_id: str = "entra-user",
    properties: tuple[str, ...] = ("115r",),
) -> AuthorizationContext:
    return AuthorizationContext(
        user=user_for(role, user_id=user_id),
        allowed_property_codes=properties,
        property_code="115r",
    )


def executor_for(permission: ToolPermission, trace: list[dict] | None = None) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="secured_tool",
            description="Test the deterministic authorization boundary.",
            input_model=EmptyInput,
            output_model=AllowedOutput,
            required_permission=permission,
        ),
        lambda _input, _context: {"allowed": True},
    )
    return ToolExecutor(registry, trace_sink=(trace if trace is not None else []).append)


def tool_context(
    role: Role,
    *,
    user_id: str = "entra-user",
    property_code: str = "115r",
) -> TrustedToolContext:
    return TrustedToolContext(
        property_code=property_code,
        user_id=user_id,
        tenant_id="tenant-1",
        roles=(role,),
        allowed_property_codes=("115r",),
        run_id="run-1",
    )


class AuthenticationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)
        self.entra_settings = Settings(
            _env_file=None,
            auth_mode="entra",
            entra_tenant_id="00000000-0000-0000-0000-000000000001",
            entra_api_audience="00000000-0000-0000-0000-000000000002",
            auth_property_access={"entra-user": ["115r"]},
        )

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_unauthenticated_protected_request_is_rejected(self) -> None:
        app.dependency_overrides[get_settings] = lambda: self.entra_settings
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "unauthenticated")

    def test_invalid_token_is_rejected_in_entra_mode(self) -> None:
        app.dependency_overrides[get_settings] = lambda: self.entra_settings
        with patch("app.core.auth.token_validator") as validator_factory:
            validator_factory.return_value.validate.side_effect = AuthenticationError("bad token")
            response = self.client.get(
                "/models",
                headers={"Authorization": "Bearer invalid-token"},
            )
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("invalid-token", response.text)

    def test_local_development_mode_still_works(self) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "local-user")
        self.assertEqual(response.json()["role"], "PropertyManager")

    def test_local_demo_identity_switch_is_backend_issued_and_deterministic(self) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
        selected = self.client.post(
            "/auth/demo-identity",
            json={"role": "Analyst"},
        )
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.json()["user_id"], "local-demo-analyst")
        self.assertEqual(selected.json()["display_name"], "Demo Analyst")
        self.assertEqual(selected.json()["role"], "Analyst")
        self.assertIn("HttpOnly", selected.headers["set-cookie"])

        current = self.client.get("/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user_id"], "local-demo-analyst")
        self.assertEqual(current.json()["role"], "Analyst")

    def test_local_demo_identity_switch_rejects_extra_trusted_state(self) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
        response = self.client.post(
            "/auth/demo-identity",
            json={
                "role": "Analyst",
                "user_id": "attacker",
                "property_permissions": ["*"],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_local_demo_identity_switch_is_unavailable_in_entra_mode(self) -> None:
        app.dependency_overrides[get_settings] = lambda: self.entra_settings
        response = self.client.post(
            "/auth/demo-identity",
            json={"role": "PropertyManager"},
        )
        self.assertEqual(response.status_code, 404)

    def test_unauthorized_property_is_rejected_before_agent_execution(self) -> None:
        app.dependency_overrides[get_settings] = lambda: self.entra_settings
        app.dependency_overrides[get_authenticated_user] = lambda: user_for(Role.VIEWER)
        with patch("app.main.AgentRuntime") as runtime:
            response = self.client.post(
                "/chat",
                json={"property_code": "176r", "message": "Show the property"},
            )
        self.assertEqual(response.status_code, 403)
        runtime.assert_not_called()

    def test_role_supplied_by_browser_is_not_trusted(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "property_code": "115r",
                "message": "Show occupancy",
                "role": "PropertyManager",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_production_rejects_local_or_incomplete_entra_auth(self) -> None:
        with self.assertRaises(ValueError):
            Settings(_env_file=None, app_env="production", app_reload=False)
        with self.assertRaises(ValueError):
            Settings(
                _env_file=None,
                app_env="production",
                app_reload=False,
                auth_mode="entra",
            )


class EntraTokenValidatorTests(unittest.TestCase):
    def test_valid_signed_claims_become_typed_identity_without_network(self) -> None:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        tenant_id = "00000000-0000-0000-0000-000000000001"
        audience = "00000000-0000-0000-0000-000000000002"
        issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        token = jwt.encode(
            {
                "aud": audience,
                "exp": int(time.time()) + 300,
                "iss": issuer,
                "oid": "entra-user",
                "tid": tenant_id,
                "ver": "2.0",
                "name": "Ada Analyst",
                "preferred_username": "ada@example.test",
                "scp": "access_as_user",
                "roles": ["Analyst"],
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )
        validator = EntraTokenValidator(
            tenant_id=tenant_id,
            audience=audience,
            metadata_fetcher=lambda _url: {"issuer": issuer, "jwks_uri": "https://jwks.test"},
        )
        validator._jwks_client = SimpleNamespace(
            get_signing_key_from_jwt=lambda _token: SimpleNamespace(key=private_key.public_key())
        )

        identity = validator.validate(token)

        self.assertEqual(identity.user_id, "entra-user")
        self.assertEqual(identity.primary_role, Role.ANALYST)
        self.assertEqual(identity.email, "ada@example.test")


class ToolAuthorizationTests(unittest.TestCase):
    def test_viewer_can_use_retrieval(self) -> None:
        result = executor_for(ToolPermission.RETRIEVAL_READ).execute(
            "secured_tool",
            {},
            tool_context(Role.VIEWER),
        )
        self.assertEqual(result.status, "succeeded")

    def test_analyst_can_execute_analytical_tools(self) -> None:
        result = executor_for(ToolPermission.ANALYTICS_READ).execute(
            "secured_tool",
            {},
            tool_context(Role.ANALYST),
        )
        self.assertEqual(result.status, "succeeded")

    def test_tool_arguments_cannot_override_authenticated_identity(self) -> None:
        captured: dict[str, object] = {}
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="secured_tool",
                description="Capture trusted context.",
                input_model=EmptyInput,
                output_model=AllowedOutput,
                required_permission=ToolPermission.RETRIEVAL_READ,
            ),
            lambda _input, context: (
                captured.update(user_id=context.user_id) or {"allowed": True}
            ),
        )
        result = ToolExecutor(registry).execute(
            "secured_tool",
            {
                "user_id": "attacker",
                "role": "PropertyManager",
                "roles": ["PropertyManager"],
            },
            tool_context(Role.VIEWER, user_id="trusted-user"),
        )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(captured["user_id"], "trusted-user")

    def test_llm_cannot_invent_execute_sql_or_elevate_with_role_argument(self) -> None:
        result = executor_for(ToolPermission.RETRIEVAL_READ).execute(
            "execute_sql",
            {"role": "PropertyManager"},
            tool_context(Role.VIEWER, user_id="trusted-viewer"),
        )
        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.type, "unknown_tool")

    def test_tool_arguments_cannot_override_property_scope(self) -> None:
        captured: dict[str, object] = {}
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="secured_tool",
                description="Capture trusted property.",
                input_model=EmptyInput,
                output_model=AllowedOutput,
                required_permission=ToolPermission.RETRIEVAL_READ,
            ),
            lambda _input, context: (
                captured.update(property_code=context.property_code) or {"allowed": True}
            ),
        )
        result = ToolExecutor(registry).execute(
            "secured_tool",
            {"property_code": "176r"},
            tool_context(Role.VIEWER, property_code="115r"),
        )
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(captured["property_code"], "115r")


class SqlApprovalAuthorizationTests(unittest.TestCase):
    def _waiting_run(self, user_id: str = "approval-user"):
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="Review this SQL.",
            components=[
                {
                    "type": "sql_approval",
                    "title": "Review SQL",
                    "data": {
                        "sql": "SELECT stored_sql",
                        "question": "Custom metric",
                        "model": "mock:test",
                        "status": "pending_approval",
                    },
                }
            ],
        )
        settings = Settings(_env_file=None)
        store = RecordingRunStore()
        runtime = AgentRuntime(
            settings,
            workflow_factory=lambda _: FakeWorkflow(response),
            run_store_factory=lambda _: store,
        )
        waiting = runtime.answer(
            property_code="115r",
            message="Custom metric",
            model="mock:test",
            conversation_id="conversation-1",
            authorization_context=authorization_for(
                Role.PROPERTY_MANAGER,
                user_id=user_id,
            ),
        )
        return settings, store, waiting

    def _resolve_as(self, role: Role, *, execute_sql=None):
        settings, store, waiting = self._waiting_run()
        runtime = AgentRuntime(
            settings,
            run_store_factory=lambda _: store,
            sql_executor=execute_sql or (lambda _settings, sql, _property: (sql, [])),
        )
        response = runtime.resolve_sql_approval(
            run_id=str(waiting.run_id),
            property_code="115r",
            approved=True,
            conversation_id="conversation-1",
            authorization_context=authorization_for(role, user_id="approval-user"),
        )
        return store, response

    def test_viewer_cannot_approve_sql(self) -> None:
        with self.assertRaises(AuthorizationDeniedError):
            self._resolve_as(Role.VIEWER)

    def test_analyst_cannot_approve_sql(self) -> None:
        with self.assertRaises(AuthorizationDeniedError):
            self._resolve_as(Role.ANALYST)

    def test_analyst_reaches_validated_sql_path_but_receives_denied_card(self) -> None:
        question = (
            "Calculate the average outstanding balance for occupied units with balances "
            "above the property's overall average."
        )
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="A validated draft is ready.",
            components=[
                {
                    "type": "sql_approval",
                    "title": "Review SQL",
                    "data": {
                        "sql": "SELECT server_validated_sql",
                        "question": question,
                        "model": "mock:test",
                        "status": "pending_approval",
                    },
                }
            ],
            tool_results={"sql_draft": {"sql": "SELECT server_validated_sql"}},
        )
        store = RecordingRunStore()
        runtime = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: FakeWorkflow(response),
            run_store_factory=lambda _: store,
        )

        denied = runtime.answer(
            property_code="115r",
            message=question,
            model="mock:test",
            conversation_id="conversation-analyst",
            authorization_context=authorization_for(
                Role.ANALYST,
                user_id="analyst-user",
            ),
        )

        card = denied.components[0].data
        self.assertEqual(denied.run_status, "completed")
        self.assertEqual(card["authorization"], "denied")
        self.assertEqual(card["current_role"], "Analyst")
        self.assertEqual(card["required_role"], "PropertyManager")
        self.assertFalse(card["executable"])
        self.assertNotIn("sql", card)
        self.assertNotIn("sql_draft", denied.tool_results)
        self.assertIn("requires PropertyManager permission", denied.answer_markdown)
        self.assertIsNone(store.saved[-1]["pending_approval"])
        self.assertFalse(
            any(event.get("tool_name") == "execute_approved_sql" for event in store.events)
        )
        event_types = [event["event_type"] for event in store.events]
        self.assertIn("approval_requested", event_types)
        self.assertIn("SQL_APPROVAL_DENIED", event_types)
        denial = next(
            event for event in store.events if event["event_type"] == "SQL_APPROVAL_DENIED"
        )
        self.assertEqual(denial["payload"]["role"], "Analyst")
        self.assertEqual(denial["payload"]["permission"], "sql.approve")
        self.assertEqual(denial["payload"]["outcome"], "denied")

    def test_property_manager_can_approve_server_stored_sql(self) -> None:
        calls: list[str] = []

        def execute_sql(_settings, sql: str, _property: str):
            calls.append(sql)
            return sql, [{"result": 1}]

        store, response = self._resolve_as(Role.PROPERTY_MANAGER, execute_sql=execute_sql)
        self.assertEqual(response.run_status, "completed")
        self.assertEqual(calls, ["SELECT stored_sql"])
        event_types = [event["event_type"] for event in store.events]
        self.assertIn("SQL_APPROVAL_AUTHORIZED", event_types)
        self.assertIn("approval_received", event_types)
        self.assertIn("evidence_recorded", event_types)
        self.assertIn("verification_succeeded", event_types)

    def test_approval_authorization_is_rechecked_at_decision_time(self) -> None:
        calls: list[str] = []
        with self.assertRaises(AuthorizationDeniedError):
            self._resolve_as(
                Role.ANALYST,
                execute_sql=lambda _settings, sql, _property: (calls.append(sql) or sql, []),
            )
        self.assertEqual(calls, [])

    def test_approval_rechecks_actor_identity(self) -> None:
        settings, store, waiting = self._waiting_run(user_id="original-manager")
        calls: list[str] = []
        runtime = AgentRuntime(
            settings,
            run_store_factory=lambda _: store,
            sql_executor=lambda _settings, sql, _property: (calls.append(sql) or sql, []),
        )
        with self.assertRaises(AgentRunNotFoundError):
            runtime.resolve_sql_approval(
                run_id=str(waiting.run_id),
                property_code="115r",
                approved=True,
                conversation_id="conversation-1",
                authorization_context=authorization_for(
                    Role.PROPERTY_MANAGER,
                    user_id="different-manager",
                ),
            )
        self.assertEqual(calls, [])

    def test_property_scope_cannot_change_during_approval(self) -> None:
        settings, store, waiting = self._waiting_run()
        calls: list[str] = []
        runtime = AgentRuntime(
            settings,
            run_store_factory=lambda _: store,
            sql_executor=lambda _settings, sql, _property: (calls.append(sql) or sql, []),
        )
        with self.assertRaises(AgentRunNotFoundError):
            runtime.resolve_sql_approval(
                run_id=str(waiting.run_id),
                property_code="176r",
                approved=True,
                conversation_id="conversation-1",
                authorization_context=authorization_for(
                    Role.PROPERTY_MANAGER,
                    user_id="approval-user",
                    properties=("115r", "176r"),
                ).for_property("176r"),
            )
        self.assertEqual(calls, [])

    def test_denial_is_a_sanitized_durable_run_event(self) -> None:
        settings, store, waiting = self._waiting_run()
        runtime = AgentRuntime(settings, run_store_factory=lambda _: store)
        with self.assertRaises(AuthorizationDeniedError):
            runtime.resolve_sql_approval(
                run_id=str(waiting.run_id),
                property_code="115r",
                approved=True,
                conversation_id="conversation-1",
                authorization_context=authorization_for(
                    Role.VIEWER,
                    user_id="approval-user",
                ),
            )
        denial = next(
            event for event in store.events if event["event_type"] == "SQL_APPROVAL_DENIED"
        )
        self.assertEqual(denial["payload"]["permission"], "sql.approve")
        self.assertEqual(denial["payload"]["outcome"], "denied")
        self.assertNotIn("sql", denial["payload"])
        self.assertNotIn("token", denial["payload"])


if __name__ == "__main__":
    unittest.main()
