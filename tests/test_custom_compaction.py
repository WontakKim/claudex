"""Registered compaction destinations through config and gateway endpoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from claudex.cli import admin_client, compact
from claudex.config import (
    AnthropicCompatibleProvider,
    ConfigError,
    GatewayConfig,
    OpenAICompatibleProvider,
)
from claudex.providers.anthropic_compatible_client import AnthropicCompatibleClient
from claudex.providers.openai_compatible_client import OpenAICompatibleClient
from test_relay import _compaction_body, _gateway, _TrackedByteStream


@pytest.mark.parametrize("family,wire_api", [("openai_compatible", "responses"), ("anthropic_compatible", "messages")])
def test_compaction_load_accepts_registered_destination_and_preserves_model_colons(tmp_path, family, wire_api):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "custom_providers": {family: {"summary": {"base_url": "https://summary.invalid/v1", **({"wire_api": wire_api} if family == "openai_compatible" else {})}}},
        "compaction.model": "summary:vendor:model",
    }))
    assert GatewayConfig.load(path).compaction_model == "summary:vendor:model"

def _custom_gateway(protocol, handler, **kwargs):
    if protocol == "responses":
        provider = OpenAICompatibleProvider(
            "responses", "https://summary.invalid/v1", "target-secret"
        )
    else:
        provider = AnthropicCompatibleProvider(
            "https://summary.invalid/v1", "target-secret"
        )
    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = (OpenAICompatibleClient("summary", provider, transport) if protocol == "responses"
              else AnthropicCompatibleClient("summary", provider, transport))
    config = GatewayConfig(
        model_map={"opus": "codex:source"},
        compaction_model="summary:vendor:model",
        custom_providers={"summary": provider},
        context_window_map={"summary:vendor:model": 1},
        **kwargs,
    )
    return _gateway(config, lambda request: pytest.fail("native Anthropic must not be called"),
                    custom_provider_clients={"summary": target}, codex_context_window=1)


def _successful_response(protocol, stream=False):
    if protocol == "responses":
        events = [{"type": "response.created", "response": {"id": "resp_summary", "model": "vendor:model"}},
                  {"type": "response.completed", "response": {"id": "resp_summary", "output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}]
        return httpx.Response(200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
    if stream:
        return httpx.Response(200, text='event: message_start\ndata: {"type":"message_start","message":{"model":"vendor:model"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n')
    return httpx.Response(200, json={"id": "msg_summary", "type": "message", "model": "vendor:model", "content": []})


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
def test_custom_compaction_uses_registered_transport_without_client_credentials(protocol, stream):
    captured = []
    def handler(request):
        captured.append(request)
        return _successful_response(protocol, stream)
    client, source = _custom_gateway(protocol, handler)
    body = _compaction_body("claude-opus", thinking_block=True)
    body["stream"] = stream
    response = client.post("/v1/messages", json=body, headers={"x-api-key": "client-secret"})
    assert response.status_code == 200
    assert len(captured) == 1
    assert str(captured[0].url) == "https://summary.invalid/v1/" + protocol
    assert captured[0].headers["authorization"] == "Bearer target-secret"
    assert "x-api-key" not in captured[0].headers
    outgoing = json.loads(captured[0].content)
    assert outgoing["model"] == "vendor:model"
    assert "sig-from-a-different-backend" not in captured[0].content.decode()
    assert not source.payloads
    record = client.app.state.compaction_last_reroute
    assert record["target_model"] == "summary:vendor:model"
    assert record["mapped_model"] == "codex:source"
    assert record["outcome"] == "rerouted"

@pytest.mark.parametrize("value", ["missing:model", "summary: ", "summary:bad model", "claude:summary:model"])
def test_invalid_stored_compaction_target_is_actionable(tmp_path, value):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"compaction.model": value}))
    with pytest.raises(ConfigError, match="compaction"):
        GatewayConfig.load(path)


def test_cli_validates_membership_after_loading_and_before_probe(monkeypatch, tmp_path, capsys):
    config = GatewayConfig(settings_file=tmp_path / "settings.json", custom_providers={
        "summary": AnthropicCompatibleProvider("https://summary.invalid/v1")})
    monkeypatch.setattr(admin_client.daemon, "_load_config", lambda: config)
    probes = []
    monkeypatch.setattr(admin_client, "_classify_daemon", lambda *args: probes.append(args) or admin_client.ProbeOutcome.NO_LISTENER)
    assert compact._compact_main(["set", "missing:model"]) == 2
    assert "register" in capsys.readouterr().err
    assert probes == []
    assert compact._compact_main(["set", "summary:vendor:model"]) == 0
    assert "summary:vendor:model" in config.settings_file.read_text()


def test_admin_custom_target_update_affects_actual_requests_and_persists(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
    seen = []
    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return _successful_response("messages")
    client, source = _custom_gateway("messages", handler, settings_file=tmp_path / "settings.json")
    client.app.state.admin_lock = asyncio.Lock()
    assert client.post("/v1/messages", json=_compaction_body("claude-opus")).status_code == 200
    response = client.put("http://127.0.0.1/admin/settings/compaction", json={"model": "summary:new:model"})
    assert response.status_code == 200
    assert response.json()["model"] == "summary:new:model"
    assert "summary:new:model" in (tmp_path / "settings.json").read_text()
    assert client.post("/v1/messages", json=_compaction_body("claude-opus")).status_code == 200
    assert seen == ["vendor:model", "new:model"]
    assert not source.payloads

@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wire_body", ['data: {}\n\n', 'data: {"type":"error","message":"bad request"}\n\n'])
def test_malformed_first_responses_event_falls_back_before_commit(stream, wire_body):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, text=wire_body)
    client, source = _custom_gateway("responses", handler)
    body = _compaction_body("claude-opus", thinking_block=True)
    body["stream"] = stream
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 503
    assert len(calls) == len(source.payloads) == 1
    assert client.app.state.compaction_last_reroute["detail"] == "invalid_json"


@pytest.mark.parametrize("invalid_body", [b"not-json", b"[]", b"{}", b'{"type":"message","usage":NaN}'])
def test_invalid_messages_nonstream_body_falls_back(invalid_body):
    stream = _TrackedByteStream([invalid_body])
    client, source = _custom_gateway("messages", lambda request: httpx.Response(200, stream=stream))
    assert client.post("/v1/messages", json=_compaction_body("claude-opus")).status_code == 503
    assert len(source.payloads) == 1
    assert stream.closed
    assert client.app.state.compaction_last_reroute["detail"] == "invalid_json"


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("failure", ["http", "connect", "read"])
def test_custom_failure_falls_back_once_with_original_request(protocol, failure):
    from claudex.translate import translate_claude_request_to_codex
    calls = []
    streams = []
    def handler(request):
        calls.append(request)
        if failure == "connect":
            raise httpx.ConnectError("private connection detail")
        if failure == "http":
            return httpx.Response(401, text="private upstream detail")
        stream = _TrackedByteStream([], httpx.ReadError("private read detail"))
        streams.append(stream)
        return httpx.Response(200, stream=stream)
    client, source = _custom_gateway(protocol, handler)
    body = _compaction_body("claude-opus", thinking_block=True)
    expected = translate_claude_request_to_codex(body, "source", None, codex_regex_compat=True)
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 503
    assert len(calls) == len(source.payloads) == 1
    assert source.payloads[0]["input"] == expected["input"]
    assert all(stream.closed for stream in streams)
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == {"http": "http_401", "connect": "connect_error", "read": "read_error"}[failure]
    assert "private" not in json.dumps(record)

def test_admin_switch_while_source_window_lookup_is_paused_keeps_captured_target(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
    seen = []
    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return _successful_response("messages")
    client, source = _custom_gateway("messages", handler, settings_file=tmp_path / "settings.json")

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def context_window(model):
            if not started.is_set():
                started.set()
                await release.wait()
            return 1
        source.context_window = context_window
        client.app.state.admin_lock = asyncio.Lock()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(client.app), base_url="http://127.0.0.1") as api:
            pending = asyncio.create_task(api.post("/v1/messages", json=_compaction_body("claude-opus")))
            await asyncio.wait_for(started.wait(), 2)
            try:
                updated = await api.put("/admin/settings/compaction", json={"model": "summary:new:model"})
                assert updated.status_code == 200
            finally:
                release.set()
            assert (await pending).status_code == 200
            assert (await api.post("/v1/messages", json=_compaction_body("claude-opus"))).status_code == 200
    asyncio.run(scenario())
    assert seen == ["vendor:model", "new:model"]
    assert not source.payloads


def test_identical_custom_source_and_target_executes_only_normal_route():
    seen = []
    def handler(request):
        seen.append(request)
        return _successful_response("responses")
    client, source = _custom_gateway("responses", handler)
    client.app.state.config = replace(client.app.state.config, model_map={"opus": "summary:vendor:model"})
    assert client.post("/v1/messages", json=_compaction_body("claude-opus")).status_code == 200
    assert len(seen) == 1
    assert not source.payloads
    assert client.app.state.compaction_last_reroute is None


def test_native_compatible_source_does_not_trigger_compaction():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return _successful_response("messages")
    client, _ = _custom_gateway("messages", handler)
    client.app.state.config = replace(client.app.state.config, model_map={"opus": "summary:source-model"})
    assert client.post("/v1/messages", json=_compaction_body("claude-opus")).status_code == 200
    assert seen == ["source-model"]
    assert client.app.state.compaction_last_reroute is None


@pytest.mark.parametrize("protocol", ["responses", "messages"])
def test_custom_midstream_failure_is_in_band_without_mapped_fallback(protocol):
    prefix = (b'data: {"type":"response.created","response":{"id":"r"}}\n\n' if protocol == "responses"
              else b'event: message_start\ndata: {"type":"message_start","message":{"model":"m"}}\n\n')
    stream = _TrackedByteStream([prefix], httpx.ReadError("private-stream-detail"))
    client, source = _custom_gateway(protocol, lambda request: httpx.Response(200, stream=stream))
    body = {**_compaction_body("claude-opus"), "stream": True}
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200
    assert "event: error" in response.text
    assert not source.payloads
    assert stream.closed
    assert client.app.state.compaction_last_reroute["outcome"] == "midstream_error"
    assert "private-stream-detail" not in json.dumps(client.app.state.compaction_last_reroute)


@pytest.mark.parametrize("failure", ["validation", "persistence", "environment"])
def test_failed_admin_update_keeps_live_custom_target(tmp_path, monkeypatch, failure):
    monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
    client, _ = _custom_gateway("messages", lambda request: _successful_response("messages"), settings_file=tmp_path / "settings.json")
    client.app.state.admin_lock = asyncio.Lock()
    original = client.app.state.config
    target = "missing:model" if failure == "validation" else "summary:new"
    if failure == "environment":
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
    if failure == "persistence":
        def fail(*args, **kwargs):
            raise ConfigError("disk unavailable")
        monkeypatch.setattr("claudex.admin.settings.update_settings_file", fail)
    response = client.put("http://127.0.0.1/admin/settings/compaction", json={"model": target})
    assert response.status_code == {"validation": 400, "persistence": 500, "environment": 409}[failure]
    assert client.app.state.config is original
    assert not original.settings_file.exists()

@pytest.mark.parametrize("protocol", ["responses", "messages"])
def test_older_custom_stream_error_does_not_overwrite_newer_record(protocol):
    async def scenario():
        waiting, release = asyncio.Event(), asyncio.Event()
        class PausedStream(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                if protocol == "responses":
                    yield b'data: {"type":"response.created","response":{"id":"older"}}\n\n'
                else:
                    yield b'event: message_start\ndata: {"type":"message_start","message":{"model":"old"}}\n\n'
                waiting.set()
                await release.wait()
                raise httpx.ReadError("private old error")
            async def aclose(self):
                self.closed = True
        stream = PausedStream()
        count = 0
        def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(200, stream=stream) if count == 1 else _successful_response(protocol)
        client, source = _custom_gateway(protocol, handler)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(client.app), base_url="http://127.0.0.1") as api:
            old = asyncio.create_task(api.post("/v1/messages", json={**_compaction_body("claude-opus"), "stream": True}))
            await asyncio.wait_for(waiting.wait(), 2)
            try:
                assert (await api.post("/v1/messages", json=_compaction_body("claude-opus"))).status_code == 200
                newer_record = dict(client.app.state.compaction_last_reroute)
            finally:
                release.set()
            assert "event: error" in (await old).text
        assert client.app.state.compaction_last_reroute == newer_record
        assert newer_record["outcome"] == "rerouted"
        assert not source.payloads
        assert stream.closed
    asyncio.run(scenario())


@pytest.mark.parametrize("protocol", ["responses", "messages"])
def test_disconnect_at_stream_commit_closes_custom_upstream_before_first_body(protocol):
    async def scenario():
        prefix = b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
        stream = _TrackedByteStream([prefix])
        client, source = _custom_gateway(protocol, lambda request: httpx.Response(200, stream=stream))
        body = json.dumps({**_compaction_body("claude-opus"), "stream": True}).encode()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/messages", "raw_path": b"/v1/messages", "query_string": b"",
                 "headers": [(b"content-type", b"application/json")], "server": ("127.0.0.1", 80),
                 "client": ("127.0.0.1", 1234), "root_path": ""}
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        async def send(message):
            if message["type"] == "http.response.start":
                raise asyncio.CancelledError()
            pytest.fail("body must not be sent after disconnect")
        with pytest.raises(asyncio.CancelledError):
            await client.app(scope, receive, send)
        assert stream.closed
        assert not source.payloads
    asyncio.run(scenario())


def test_native_parser_rejects_misleading_nested_provider_prefix():
    from claudex.config import parse_compaction_model
    with pytest.raises(ConfigError, match="nested provider"):
        parse_compaction_model("claude:summary:vendor-model")

@pytest.mark.parametrize("event", [
    {"type": "response.error", "error": {"message": "rejected"}},
    {"type": "response.created", "response": "invalid"},
    {"type": "response.completed", "response": {"output": "invalid"}},
])
def test_invalid_first_responses_envelope_never_commits_stream(event):
    client, source = _custom_gateway("responses", lambda request: httpx.Response(200, text="data: " + json.dumps(event) + "\n\n"))
    response = client.post("/v1/messages", json={**_compaction_body("claude-opus"), "stream": True})
    assert response.status_code == 503
    assert len(source.payloads) == 1
    assert client.app.state.compaction_last_reroute["outcome"] == "fallback_mapped"

@pytest.mark.parametrize("stream", [False, True])
def test_malformed_later_response_event_obeys_stream_commit_boundary(stream):
    events = [{"type": "response.created", "response": {"id": "r"}},
              {"type": "response.output_item.added", "item": "invalid", "output_index": 0}]
    wire = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    client, source = _custom_gateway("responses", lambda request: httpx.Response(200, text=wire))
    response = client.post("/v1/messages", json={**_compaction_body("claude-opus"), "stream": stream})
    if stream:
        assert response.status_code == 200
        assert "event: error" in response.text
        assert not source.payloads
        assert client.app.state.compaction_last_reroute["outcome"] == "midstream_error"
    else:
        assert response.status_code == 503
        assert len(source.payloads) == 1
        assert client.app.state.compaction_last_reroute["detail"] == "invalid_json"


@pytest.mark.parametrize(
    "event_field, payload, expected_outcome",
    [
        (
            "event: content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {
                "type": "text_delta",
                "text": "The log contains event: error but the summary succeeded.",
            }},
            "rerouted",
        ),
        (
            "event:error",
            {"type": "error", "error": {"type": "api_error", "message": "fixture error"}},
            "midstream_error",
        ),
        (
            "event: error",
            {"type": "error", "error": {"type": "api_error", "message": "fixture error"}},
            "midstream_error",
        ),
        (
            "event:  error",
            {"type": "error", "error": {"type": "api_error", "message": "fixture error"}},
            "rerouted",
        ),
    ],
)
def test_custom_messages_diagnostics_parse_sse_event_field_only(
    event_field, payload, expected_outcome,
):
    wire = (
        event_field + "\ndata: " + json.dumps(payload) + "\n\n"
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    ).encode()
    stream = _TrackedByteStream([wire[:7], wire[7:]])
    client, source = _custom_gateway(
        "messages", lambda request: httpx.Response(200, stream=stream)
    )
    response = client.post(
        "/v1/messages", json={**_compaction_body("claude-opus"), "stream": True}
    )
    assert response.status_code == 200
    assert response.content == wire
    assert not source.payloads
    assert stream.closed
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == expected_outcome
    assert record["detail"] == (
        "read_error" if expected_outcome == "midstream_error" else None
    )


@pytest.mark.parametrize("provider", ["codex", "kimi", "grok"])
def test_builtin_compaction_target_config_cli_and_admin(provider, tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
    target = provider + ":MixedCase:Revision"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"compaction.model": target}))
    config = GatewayConfig.load(path)
    assert config.compaction_model == target
    monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", target)
    assert GatewayConfig.from_env().compaction_model == target
    monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL")
    monkeypatch.setattr(admin_client.daemon, "_load_config", lambda: config)
    monkeypatch.setattr(admin_client, "_classify_daemon", lambda *args: admin_client.ProbeOutcome.NO_LISTENER)
    assert compact._compact_main(["set", target]) == 0
    client, _ = _gateway(config, lambda request: pytest.fail("no inference on save"))
    client.app.state.admin_lock = asyncio.Lock()
    response = client.put("http://127.0.0.1/admin/settings/compaction", json={"model": target})
    assert response.status_code == 200
    assert response.json()["model"] == target
    assert GatewayConfig.load(path).compaction_model == target


@pytest.mark.parametrize("provider", ["codex", "kimi", "grok"])
@pytest.mark.parametrize("stream", [False, True])
def test_builtin_compaction_uses_bound_client_auth_and_transport(provider, stream, monkeypatch):
    from claudex import server
    from claudex.providers.codex_client import CodexClient
    from claudex.providers.grok_client import GrokClient
    from claudex.providers.kimi_client import KimiClient
    from test_relay import AvailableCodexAuthManager, AvailableGrokAuthManager, AvailableKimiAuthManager

    captured = []
    protocol = "messages" if provider == "kimi" else "responses"

    def handler(request):
        captured.append(request)
        return _successful_response(protocol, stream)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    if provider == "codex":
        target = CodexClient(AvailableCodexAuthManager(), transport)
    elif provider == "grok":
        target = GrokClient(AvailableGrokAuthManager(), transport)
    else:
        target = KimiClient(AvailableKimiAuthManager(), transport)
    if provider != "kimi":
        async def version():
            return "0.157.1"
        monkeypatch.setattr(target._version_discovery, "get_version", version)
    source_provider = "grok" if provider == "codex" else "codex"
    config = GatewayConfig(
        model_map={"opus": source_provider + ":source"},
        compaction_model=provider + ":vendor:model",
        context_window_map={source_provider + ":source": 1, provider + ":vendor:model": 1},
    )
    client, source = _gateway(config, lambda request: pytest.fail("native destination must not be called"))
    app = client.app
    clients = {"codex": source, "kimi": app.state.kimi_client, "grok": app.state.grok_client}
    clients[provider] = target
    app.state.route_backends = server._assemble_route_backends(
        app, config, clients["codex"], clients["kimi"], clients["grok"], {})
    body = _compaction_body("claude-opus", thinking_block=True)
    body["stream"] = stream
    response = client.post("/v1/messages", json=body, headers={
        "x-api-key": "client-secret", "authorization": "Bearer client-bearer"})
    assert response.status_code == 200
    assert len(captured) == 1
    outgoing = captured[0]
    assert outgoing.url.path.endswith("/" + protocol)
    expected_token = {"codex": "codex", "kimi": "kimi-token", "grok": "grok-token"}[provider]
    assert outgoing.headers["authorization"] == "Bearer " + expected_token
    assert "x-api-key" not in outgoing.headers
    assert "client-secret" not in str(outgoing.headers)
    assert "client-bearer" not in str(outgoing.headers)
    assert json.loads(outgoing.content)["model"] == "vendor:model"
    assert "sig-from-a-different-backend" not in outgoing.content.decode()
    assert not source.payloads
    assert app.state.compaction_last_reroute["outcome"] == "rerouted"
    assert app.state.compaction_last_reroute["target_model"] == provider + ":vendor:model"
