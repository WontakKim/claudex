"""Shared request-serving support for the gateway application."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from claudex_gateway import paths
from claudex_gateway.claude_account_profile import load_account_profile_fingerprint
from claudex_gateway.claude_accounts import AccountRegistryError, mark_account_needs_reauth
from claudex_gateway.claude_ambient_account import AmbientAccountProvider, AmbientClaudeAuthManager
from claudex_gateway.claude_auth import (
    ClaudeAccountAuthManager,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.config import GatewayConfig
from claudex_gateway.usage import _provider_result, fetch_claude_account_usage

logger = logging.getLogger("claudex_gateway.server")


def _claude_error_body(error_type: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _openai_error_body(
    error_type: str, message: str, code: str | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _upstream_error_message(body: str) -> str:
    message = body
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("error")
            if isinstance(detail, dict) and detail.get("message"):
                message = str(detail["message"])
            elif isinstance(parsed.get("detail"), str):
                message = parsed["detail"]
    return message


def _require_local_token(
    request: Request, *, claude_error: bool = False
) -> JSONResponse | None:
    config: GatewayConfig = request.app.state.config
    expected_token = config.local_token
    if expected_token is None:
        return None
    authorization = request.headers.get("authorization", "")
    scheme, separator, provided_token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(provided_token, expected_token)
    ):
        body = (
            _claude_error_body("authentication_error", "Missing or invalid bearer token")
            if claude_error
            else _openai_error_body(
                "authentication_error",
                "Missing or invalid bearer token",
                "invalid_api_key",
            )
        )
        return JSONResponse(
            body,
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


def _account_profile_fingerprint(app_state: Any, account_id: str) -> str | None:
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        member = provider.pool_member()
        return member.profile_fingerprint if member is not None else None
    return load_account_profile_fingerprint(paths.accounts_dir("claude") / account_id)


async def _mark_account_needs_reauth_best_effort(app_state: Any, account_id: str) -> None:
    """Persist needs-reauth for `account_id`, never failing the caller.

    A concurrent `account remove` (AccountNotFoundError) or a registry I/O
    problem must not mask the response the caller is about to return.
    """
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        logger.debug("not marking ambient claude account %s needs-reauth", account_id)
        return
    try:
        await asyncio.to_thread(mark_account_needs_reauth, account_id)
    except AccountRegistryError as exc:
        logger.warning("could not mark claude account %s needs-reauth: %s", account_id, exc)


def _claude_account_auth_manager(
    app_state: Any, account_id: str
) -> ClaudeAccountAuthManager | AmbientClaudeAuthManager:
    """Return the cached per-account manager, creating it on first use.

    Managers are keyed by account id so a `use`-switch mid-flight gets a
    fresh manager for the new directory while requests still draining on the
    old account keep theirs. No lock is needed: there is no await between
    the lookup and the insert.
    """
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        return provider.auth_manager()
    managers: dict[str, ClaudeAccountAuthManager] = app_state.claude_account_auth_managers
    manager = managers.get(account_id)
    if manager is None:
        manager = ClaudeAccountAuthManager(
            paths.accounts_dir("claude") / account_id, app_state.http_client
        )
        managers[account_id] = manager
    return manager


def _account_usage_fetch(app_state: Any) -> Any:
    """The usage cache's fetch closure: account id -> (result, retry_after).

    Resolves the per-account auth manager lazily and converts a conclusive
    invalid_grant into a registry needs-reauth mark plus an "unavailable"
    envelope — the cache itself must never see that exception.
    """

    async def fetch(account_id: str) -> tuple[dict[str, Any], float | None]:
        manager = _claude_account_auth_manager(app_state, account_id)
        try:
            return await fetch_claude_account_usage(app_state.http_client, manager)
        except ClaudeAccountReauthRequiredError:
            await _mark_account_needs_reauth_best_effort(app_state, account_id)
            return (
                _provider_result(
                    "claude",
                    status="unavailable",
                    error="account needs re-authentication; log in again from the dashboard",
                ),
                None,
            )

    return fetch
