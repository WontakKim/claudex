"""Starlette composition root for the local multi-provider Claude Code gateway."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from claudex import paths, server_support
from claudex.claude.account_usage_cache import ClaudeAccountUsageCache
from claudex.claude.quota_429 import Quota429IncidentWriter
from claudex.admin.accounts import (
    _handle_admin_claude_account_delete,
    _handle_admin_claude_accounts_get,
    _handle_admin_claude_accounts_usage,
    _handle_admin_claude_local_get,
    _handle_admin_claude_login_code_post,
    _handle_admin_claude_login_delete,
    _handle_admin_claude_login_get,
    _handle_admin_claude_login_post,
    _handle_admin_claude_login_replace_post,
    _handle_admin_claude_pool_status,
)
from claudex.admin.common import _handle_health, _handle_hello
from claudex.admin.settings import (
    _handle_admin_claude_routing_get,
    _handle_admin_claude_routing_put,
    _handle_admin_claude_serving_delete,
    _handle_admin_claude_serving_get,
    _handle_admin_claude_serving_put,
    _handle_admin_codex_get,
    _handle_admin_codex_put,
    _handle_admin_compaction_get,
    _handle_admin_compaction_put,
    _handle_admin_log_level_get,
    _handle_admin_log_level_put,
    _handle_admin_mapping_get,
    _handle_admin_mapping_put,
)
from claudex.admin.system import (
    _handle_admin_codex_models,
    _handle_admin_codex_reset_credit,
    _handle_admin_connection_test,
    _handle_admin_custom_models,
    _handle_admin_grok_models,
    _handle_admin_kimi_models,
    _handle_admin_logs,
    _handle_admin_usage,
    _handle_dashboard,
    _handle_dashboard_css,
    _handle_dashboard_js,
    _handle_favicon,
)
from claudex.relay import endpoints as relay_endpoints
from claudex.relay.kimi import _kimi_error_to_claude, _kimi_request_headers
from claudex.claude.account_pool import AccountCooldownTracker
from claudex.claude.accounts import AccountRecord, list_accounts
from claudex.claude.ambient_account import AmbientAccountProvider, is_duplicate_identity
from claudex.balanced.polling import UsagePollAccount
from claudex.balanced.runtime import ClaudeBalancedRuntime
from claudex.providers.codex_auth import CodexAuthError, CodexAuthManager
from claudex.providers.backends import AnthropicBackend, ResponsesBackend, RouteBackend
from claudex.providers.codex_client import (
    CODEX_FAST_TIER_WIRE_VALUE,
    CodexClient,
)
from claudex.config import GatewayConfig
from claudex.providers.grok_auth import GrokAuthError, GrokAuthManager
from claudex.providers.grok_client import GrokClient, sanitize_grok_payload
from claudex.providers.kimi_auth import KimiAuthError, KimiAuthManager
from claudex.providers.kimi_client import KimiClient
from claudex.locking import try_file_lock
from claudex.providers.openai_compatible_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

_UPSTREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=15.0)


async def _adapt_grok_payload(
    payload: dict[str, Any], model: str
) -> dict[str, Any]:
    return sanitize_grok_payload(payload, model)


async def _adapt_identity_payload(
    payload: dict[str, Any], model: str
) -> dict[str, Any]:
    return payload


def _adapt_identity_probe_payload(
    payload: dict[str, Any], model: str
) -> dict[str, Any]:
    return payload


def _assemble_route_backends(
    app: Starlette,
    config: GatewayConfig,
    codex_client: CodexClient,
    kimi_client: KimiClient,
    grok_client: GrokClient,
    custom_provider_clients: dict[str, OpenAICompatibleClient],
) -> dict[str, RouteBackend]:
    async def adapt_codex_payload(
        payload: dict[str, Any], model: str
    ) -> dict[str, Any]:
        current_config: GatewayConfig = app.state.config
        if current_config.codex_service_tier != "fast":
            return payload
        if await codex_client.supports_fast_tier(model):
            payload["service_tier"] = CODEX_FAST_TIER_WIRE_VALUE
        else:
            logger.debug(
                "fast tier requested but the codex catalog does not advertise it for %s",
                model,
            )
        return payload

    route_backends: dict[str, RouteBackend] = {
        "codex": ResponsesBackend(
            transport=codex_client,
            adapt_payload=adapt_codex_payload,
            adapt_probe_payload=_adapt_identity_probe_payload,
            signature_namespace=None,
        ),
        "kimi": AnthropicBackend(
            transport=kimi_client,
            header_policy=_kimi_request_headers,
            error_policy=_kimi_error_to_claude,
            token_counter=kimi_client.count_tokens,
            catalog_loader=kimi_client.list_models,
        ),
        "grok": ResponsesBackend(
            transport=grok_client,
            adapt_payload=_adapt_grok_payload,
            adapt_probe_payload=sanitize_grok_payload,
            signature_namespace=None,
        ),
    }
    route_backends.update(
        {
            name: ResponsesBackend(
                transport=client,
                adapt_payload=_adapt_identity_payload,
                adapt_probe_payload=_adapt_identity_probe_payload,
                signature_namespace=name,
            )
            for name, client in custom_provider_clients.items()
        }
    )

    configured_provider_names = set(config.route_providers)
    client_provider_names = {"codex", "kimi", "grok", *custom_provider_clients}
    backend_provider_names = set(route_backends)
    if (
        client_provider_names != configured_provider_names
        or backend_provider_names != configured_provider_names
    ):
        raise RuntimeError(
            "route backend registry mismatch: "
            f"configured={sorted(configured_provider_names)!r}; "
            f"clients={sorted(client_provider_names)!r}; "
            f"backends={sorted(backend_provider_names)!r}"
        )
    return route_backends


class _LogBufferHandler(logging.Handler):
    """Keeps the most recent gateway log records for the dashboard's Log tab."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info and record.exc_info != (None, None, None):
                message += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            self.records.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": message,
                }
            )
        except Exception:  # A logging handler must never propagate errors.
            self.handleError(record)


def create_app(config: GatewayConfig, daemon_nonce: str | None = None) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # Every daemon capable of serving the Claude account pool — disabled,
        # fallback, or balanced routing mode alike — takes this lease before
        # any endpoint is exposed and holds it for the process lifetime.
        # Routing-mode transitions never acquire or release it. This is the
        # same nonblocking exclusive lock used to serialize `account login`.
        pool_dir = paths.claude_account_pool_dir()
        pool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        claude_pool_lease = try_file_lock(paths.claude_account_pool_lock())
        if claude_pool_lease is None:
            raise RuntimeError(
                "claude account pool is already served by another process (balanced-router.lock held)"
            )
        app.state.claude_pool_lease = claude_pool_lease
        app.state.claude_quota_429_incident_writer = Quota429IncidentWriter(
            pool_dir / "claude-429-incidents.jsonl"
        )
        try:
            log_buffer = _LogBufferHandler()
            logging.getLogger().addHandler(log_buffer)
            app.state.log_buffer = log_buffer
            try:
                async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as http_client:
                    codex_auth_manager = CodexAuthManager(config.codex_home / "auth.json", http_client)
                    kimi_auth_manager = KimiAuthManager(config.kimi_code_home, http_client)
                    grok_auth_manager = GrokAuthManager(config.grok_home / "auth.json", http_client)
                    app.state.config = config
                    app.state.admin_lock = asyncio.Lock()
                    # Redeem key of a reset attempt whose outcome never came back.
                    app.state.codex_reset_key = None
                    # Diagnostics for the compaction reroute (see
                    # _assign_compaction_reroute): no reroute has been attempted
                    # yet, and the sequence counter starts a fresh count for this
                    # process's lifetime.
                    app.state.compaction_last_reroute = None
                    app.state.compaction_reroute_sequence = 0
                    app.state.codex_auth_manager = codex_auth_manager
                    app.state.codex_client = CodexClient(codex_auth_manager, http_client)
                    app.state.kimi_auth_manager = kimi_auth_manager
                    app.state.kimi_client = KimiClient(kimi_auth_manager, http_client)
                    app.state.grok_auth_manager = grok_auth_manager
                    app.state.grok_client = GrokClient(grok_auth_manager, http_client)
                    custom_provider_clients: dict[str, OpenAICompatibleClient] = {}
                    for name, provider in config.custom_providers.items():
                        custom_provider_clients[name] = OpenAICompatibleClient(
                            name, provider, http_client
                        )
                    app.state.custom_provider_clients = custom_provider_clients
                    app.state.route_backends = _assemble_route_backends(
                        app,
                        config,
                        app.state.codex_client,
                        app.state.kimi_client,
                        app.state.grok_client,
                        custom_provider_clients,
                    )
                    app.state.http_client = http_client

                    if config.claude_account_routing_mode == "balanced":
                        # Restore a persisted balanced setting only after the pool lease
                        # is held. A request racing this window sees `status ==
                        # "acquiring"`, waits for the transition, and then re-reads the
                        # published mode. `persist` is a no-op because the setting is
                        # already on disk.
                        try:
                            await app.state.claude_balanced_runtime.prepare_and_publish(
                                accounts=list_accounts(),
                                accounts_root=paths.accounts_dir("claude"),
                                runtime_db_path=paths.claude_account_pool_runtime_db(),
                                persist=lambda: None,
                                entry="startup_restore",
                                usage_cache=app.state.claude_account_usage_cache,
                            )
                            logger.info(
                                "balanced routing runtime restored (epoch=%s)",
                                app.state.claude_balanced_runtime.epoch_id,
                            )
                        except Exception as exc:
                            # Degrade, don't crash the daemon: balanced dispatch fails
                            # closed (503 "balanced routing is not active") until an
                            # admin fixes the underlying issue and re-enables it.
                            logger.error(
                                "could not activate persisted balanced routing at "
                                "startup: %s",
                                exc,
                                exc_info=True,
                            )

                    try:
                        credentials = await codex_auth_manager.get_credentials()
                        logger.info(
                            "codex credentials ready (mode=%s, account=%s)",
                            "api_key" if credentials.is_api_key else "chatgpt",
                            credentials.account_id,
                        )
                    except CodexAuthError as exc:
                        logger.warning("codex direction unavailable: %s", exc)

                    if config.maps_to_provider("kimi"):
                        try:
                            await kimi_auth_manager.get_credentials()
                            logger.info("kimi credentials ready")
                        except KimiAuthError as exc:
                            logger.warning("kimi direction unavailable: %s", exc)

                    if config.maps_to_provider("grok"):
                        try:
                            await grok_auth_manager.get_credentials()
                            logger.info("grok credentials ready")
                        except GrokAuthError as exc:
                            logger.warning("grok direction unavailable: %s", exc)

                    for name, custom_client in custom_provider_clients.items():
                        if not config.maps_to_provider(name):
                            continue
                        try:
                            models = await custom_client.list_models()
                            logger.info(
                                "custom provider '%s' ready (%d models)", name, len(models)
                            )
                        except Exception as exc:
                            logger.warning("custom provider '%s' unavailable: %s", name, exc)

                    try:
                        yield
                    finally:
                        refresh_tasks = (
                            *tuple(app.state.claude_usage_refresh_tasks),
                            *tuple(app.state.claude_pin_refresh_tasks),
                        )
                        for refresh_task in refresh_tasks:
                            refresh_task.cancel()
                        if refresh_tasks:
                            await asyncio.gather(
                                *refresh_tasks, return_exceptions=True
                            )
            finally:
                logging.getLogger().removeHandler(log_buffer)
        finally:
            # Process shutdown while settings remain "balanced" preserves the
            # persisted mode, epoch id/seed, pins, observations, cooldowns, and
            # capability evidence. Unlike shutdown, an intentional `exit_mode`
            # invalidates them. Finalization must complete before the process
            # lease is released, so another process cannot open the runtime store
            # while this one is still draining or closing it.
            await app.state.claude_balanced_runtime.shutdown_preserving_epoch()
            app.state.claude_pool_lease.release()

    app = Starlette(
        routes=[
            Route("/", _handle_dashboard, methods=["GET"]),
            Route("/dashboard.css", _handle_dashboard_css, methods=["GET"]),
            Route("/dashboard.js", _handle_dashboard_js, methods=["GET"]),
            Route("/favicon.ico", _handle_favicon, methods=["GET"]),
            Route("/v1/messages", relay_endpoints._handle_messages, methods=["POST"]),
            Route(
                "/v1/messages/count_tokens",
                relay_endpoints._handle_count_tokens,
                methods=["POST"],
            ),
            Route("/api/hello", _handle_hello, methods=["GET"]),
            Route("/health", _handle_health, methods=["GET"]),
            # Admin routes use settings/* for gateway-wide settings and
            # providers/{p}/* for each backend's own surface, with top-level
            # logs/usage/test as cross-cutting
            # observability. No aliases: old paths 404.
            Route("/admin/settings/mapping", _handle_admin_mapping_get, methods=["GET"]),
            Route("/admin/settings/mapping", _handle_admin_mapping_put, methods=["PUT"]),
            Route(
                "/admin/settings/log-level", _handle_admin_log_level_get, methods=["GET"]
            ),
            Route(
                "/admin/settings/log-level", _handle_admin_log_level_put, methods=["PUT"]
            ),
            Route(
                "/admin/settings/compaction", _handle_admin_compaction_get, methods=["GET"]
            ),
            Route(
                "/admin/settings/compaction", _handle_admin_compaction_put, methods=["PUT"]
            ),
            Route("/admin/settings/codex", _handle_admin_codex_get, methods=["GET"]),
            Route("/admin/settings/codex", _handle_admin_codex_put, methods=["PUT"]),
            Route(
                "/admin/providers/codex/models",
                _handle_admin_codex_models,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/codex/reset-credit",
                _handle_admin_codex_reset_credit,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/kimi/models", _handle_admin_kimi_models, methods=["GET"]
            ),
            Route(
                "/admin/providers/grok/models", _handle_admin_grok_models, methods=["GET"]
            ),
            Route(
                "/admin/providers/custom/{name}/models",
                _handle_admin_custom_models,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/local",
                _handle_admin_claude_local_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/accounts",
                _handle_admin_claude_accounts_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/accounts/{account_id}",
                _handle_admin_claude_account_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/login/code",
                _handle_admin_claude_login_code_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/login/replace",
                _handle_admin_claude_login_replace_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_put,
                methods=["PUT"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/pool/routing",
                _handle_admin_claude_routing_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/routing",
                _handle_admin_claude_routing_put,
                methods=["PUT"],
            ),
            Route(
                "/admin/providers/claude/pool/status",
                _handle_admin_claude_pool_status,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/usage",
                _handle_admin_claude_accounts_usage,
                methods=["GET"],
            ),
            Route("/admin/logs", _handle_admin_logs, methods=["GET"]),
            Route("/admin/usage", _handle_admin_usage, methods=["GET"]),
            Route("/admin/test", _handle_admin_connection_test, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.daemon_nonce = daemon_nonce
    # Lifespan replaces this with one client per configured custom provider.
    app.state.custom_provider_clients = {}
    # Lazily-created ClaudeAccountAuthManager per registered account id.
    # Initialized here (not in the lifespan) so the dict exists even when a
    # test drives the app without entering the lifespan context.
    app.state.claude_account_auth_managers = {}
    app.state.claude_ambient_accounts = (
        AmbientAccountProvider() if config.claude_account_include_local_login else None
    )
    # Dashboard login session slot (single concurrent session) and the
    # per-account usage cache — the fetch closure resolves http_client from
    # app.state at call time, so wiring here works with or without the
    # lifespan having run.
    app.state.claude_login_session = None
    app.state.claude_account_usage_cache = ClaudeAccountUsageCache(
        fetch=server_support._account_usage_fetch(app.state)
    )
    # Strong references keep detached refreshes alive until completion.
    app.state.claude_usage_refresh_tasks = set()
    app.state.claude_pin_refresh_tasks = set()
    # Rate-limit cooldowns are ephemeral runtime state and live only in this
    # process; the registry records durable facts only.
    app.state.claude_account_cooldowns = AccountCooldownTracker()
    # Starts "disabled" — a persisted "balanced" mode is prepared and
    # published during lifespan startup after the pool lease is held. It is
    # initialized here so balanced dispatch fails closed
    # rather than crashing, even for a test that drives the app without
    # entering the lifespan context.
    app.state.claude_balanced_runtime = ClaudeBalancedRuntime()

    def _ambient_usage_poll_supplier(
        records: Sequence[AccountRecord],
    ) -> UsagePollAccount | None:
        provider: AmbientAccountProvider | None = app.state.claude_ambient_accounts
        if provider is None:
            return None
        member = provider.pool_member()
        if member is None or is_duplicate_identity(member, records):
            return None
        return UsagePollAccount(
            account_id=member.record.id,
            account_incarnation_id=member.record.account_incarnation_id,
            account_profile_fingerprint=member.profile_fingerprint,
        )

    app.state.claude_balanced_runtime.ambient_usage_poll_supplier = (
        _ambient_usage_poll_supplier
    )
    return app
