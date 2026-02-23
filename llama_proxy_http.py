#!/usr/bin/env python3
"""
HTTP proxy that exposes an OpenAI-style endpoint for Android clients and forwards
requests to the local llama.cpp HTTP server.

Default listen port is 18081 to match cloudflared ingress:
  llama.jimmyandjonny.work -> http://127.0.0.1:18081

Upstream defaults come from the local /data infra config:
  - host: /data/etc/hosts/<hostname>.json (llama_cpp.host, default 127.0.0.1)
  - port: /data/etc/ports.json (services.llama_cpp.base, default 14829)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web


@dataclass(frozen=True)
class InfraCfg:
    data_root: Path
    hostname: str
    ports_cfg_path: Path
    host_cfg_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _get_nested(d: dict[str, Any], key: str, default: Any) -> Any:
    cur: Any = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def load_infra_cfg() -> InfraCfg:
    data_root = Path(os.environ.get("INFRA_DATA_ROOT", "/data"))
    hostname = os.environ.get("INFRA_HOST", socket.gethostname())
    return InfraCfg(
        data_root=data_root,
        hostname=hostname,
        ports_cfg_path=data_root / "etc" / "ports.json",
        host_cfg_path=data_root / "etc" / "hosts" / f"{hostname}.json",
    )


def infer_upstream_base_url(infra: InfraCfg) -> str:
    host_cfg = _read_json(infra.host_cfg_path)
    ports_cfg = _read_json(infra.ports_cfg_path)

    upstream_host = os.environ.get("LLAMA_UPSTREAM_HOST") or _get_nested(host_cfg, "llama_cpp.host", "127.0.0.1")

    base_port = _get_nested(ports_cfg, "services.llama_cpp.base", 14829)
    try:
        base_port_int = int(base_port)
    except Exception:
        base_port_int = 14829

    inst = os.environ.get("LLAMA_UPSTREAM_INSTANCE")
    if inst is None:
        inst = _get_nested(host_cfg, "llama_cpp.instance", 0)
    try:
        inst_int = int(inst)
    except Exception:
        inst_int = 0

    upstream_port = int(os.environ.get("LLAMA_UPSTREAM_PORT") or (base_port_int + inst_int))
    return f"http://{upstream_host}:{upstream_port}"


def _json_error(message: str, *, status: int = 500) -> web.Response:
    body = {"error": {"message": message, "type": "proxy_error"}}
    return web.json_response(body, status=status)


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _proxy_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    json_body: Any | None = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_total_s: float = 180.0,
) -> web.Response:
    timeout = aiohttp.ClientTimeout(total=timeout_total_s, connect=10, sock_connect=10, sock_read=timeout_total_s)

    try:
        async with session.request(
            method,
            url,
            json=json_body,
            data=raw_body,
            headers=headers,
            timeout=timeout,
        ) as resp:
            data = await resp.read()
            # Preserve upstream content-type if present.
            ct = resp.headers.get("Content-Type", "application/json")
            return web.Response(status=resp.status, body=data, headers={"Content-Type": ct})
    except asyncio.TimeoutError:
        return _json_error(f"Upstream timeout calling {url}", status=504)
    except Exception as e:
        return _json_error(f"Upstream error calling {url}: {e!r}", status=502)


async def handle_health(request: web.Request) -> web.StreamResponse:
    app = request.app
    upstream = app["upstream_base_url"]
    session: aiohttp.ClientSession = app["session"]
    upstream_url = f"{upstream}/health"
    return await _proxy_request(session, "GET", upstream_url, timeout_total_s=10.0)


async def handle_models(request: web.Request) -> web.StreamResponse:
    app = request.app
    upstream = app["upstream_base_url"]
    session: aiohttp.ClientSession = app["session"]
    upstream_url = f"{upstream}/v1/models"
    return await _proxy_request(session, "GET", upstream_url, timeout_total_s=20.0)


def _apply_model_override(payload: dict[str, Any], model_override: str | None) -> None:
    if model_override is None or not model_override.strip():
        return
    payload["model"] = model_override.strip()


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    app = request.app
    upstream = app["upstream_base_url"]
    session: aiohttp.ClientSession = app["session"]
    model_override: str | None = app["model_override"]
    force_non_streaming: bool = app["force_non_streaming"]

    try:
        payload = await request.json()
    except Exception as e:
        return _json_error(f"Invalid JSON body: {e}", status=400)

    if not isinstance(payload, dict):
        return _json_error("JSON body must be an object", status=400)

    # Keep Android-side behavior simple: block until reply arrives.
    if force_non_streaming:
        payload["stream"] = False

    _apply_model_override(payload, model_override)

    upstream_url = f"{upstream}/v1/chat/completions"
    t0 = _now_ms()
    resp = await _proxy_request(session, "POST", upstream_url, json_body=payload, timeout_total_s=600.0)
    dt = _now_ms() - t0
    logging.info("POST /v1/chat/completions -> %s in %d ms", resp.status, dt)
    return resp


async def handle_root(request: web.Request) -> web.StreamResponse:
    app = request.app
    return web.json_response(
        {
            "service": "llama_proxy_http",
            "upstream": app["upstream_base_url"],
            "endpoints": ["/health", "/v1/models", "/v1/chat/completions"],
        }
    )


async def on_startup(app: web.Application) -> None:
    app["session"] = aiohttp.ClientSession()


async def on_cleanup(app: web.Application) -> None:
    session: aiohttp.ClientSession = app.get("session")
    if session is not None:
        await session.close()


def make_app(*, upstream_base_url: str, model_override: str | None, force_non_streaming: bool) -> web.Application:
    app = web.Application(client_max_size=10 * 1024 * 1024)
    app["upstream_base_url"] = upstream_base_url.rstrip("/")
    app["model_override"] = model_override
    app["force_non_streaming"] = force_non_streaming
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible HTTP proxy -> local llama.cpp")
    p.add_argument("--host", default=os.environ.get("LLAMA_PROXY_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LLAMA_PROXY_PORT", "18081")))
    p.add_argument("--upstream", default=os.environ.get("LLAMA_UPSTREAM_URL"))
    p.add_argument(
        "--model-override",
        default=os.environ.get("LLAMA_PROXY_MODEL"),
        help="If set, overwrite payload.model before forwarding upstream.",
    )
    p.add_argument(
        "--allow-stream",
        action="store_true",
        default=(os.environ.get("LLAMA_PROXY_ALLOW_STREAM", "").strip() == "1"),
        help="Allow stream=true to pass through (proxy still returns raw upstream bytes).",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("LLAMA_PROXY_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    upstream = args.upstream
    if not upstream:
        infra = load_infra_cfg()
        upstream = infer_upstream_base_url(infra)

    force_non_streaming = not bool(args.allow_stream)

    app = make_app(upstream_base_url=upstream, model_override=args.model_override, force_non_streaming=force_non_streaming)

    logging.info("Starting llama proxy on %s:%s -> %s", args.host, args.port, upstream)
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
