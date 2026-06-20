"""
Orca Web Server — FastAPI backend for the browser UI.

Endpoints:
  GET  /                    → serves the web UI
  GET  /api/status          → model, memory stats, uptime
  POST /api/chat            → single-shot response
  POST /api/stream          → SSE streaming response
  POST /api/memory/recall   → query long-term memory
  POST /api/remember        → store a fact permanently
  GET  /api/sessions        → list past sessions
  POST /api/session/load    → resume a session

100% local — no external calls from the server itself.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orca.license import current_tier, get_active_license, has_feature
from orca.license.keys import format_expiry

from orca.auth import auth_router, get_current_user_optional, check_quota, increment_usage
from orca.auth.rbac import require_permission
from orca.auth.store import User
from orca import audit

from orca.brain.providers import get_brain
from orca.brain.memory import MemoryEngine, EpisodicMemory
from orca.brain.agent import AgentLoop
from orca.brain.context import ContextManager
from orca.tools import build_registry
from orca.character import CORE_SYSTEM_WITH_TOOLS
from orca.config import CONFIG, ORCA_HOME
from orca.variants.ultra import OrcaUltra

_START_TIME = time.time()
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Atheris API", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


# ─────────────────────────────────────────────────────────────────────────────
#  Session store — one AgentLoop + MemoryEngine per browser session
# ─────────────────────────────────────────────────────────────────────────────

def _model_name_for_variant(variant: str | None) -> str:
    if variant == "nano":
        return CONFIG.ollama.model_nano
    if variant == "ultra":
        return CONFIG.ollama.model_ultra
    return CONFIG.ollama.model_core


class _Session:
    def __init__(self, session_id: str, model_variant: str | None = None):
        self.id = session_id
        self.model_variant = model_variant or "core"
        self.memory = MemoryEngine(session_id=session_id)
        brain = get_brain(_model_name_for_variant(model_variant))
        self.ctx = ContextManager(brain)
        tools = build_registry(memory_engine=self.memory)
        self.agent = AgentLoop(brain=brain, tools=tools, session_id=session_id)
        prior = self.memory.load_prior_context()
        if prior:
            self.agent.load_history([
                {"role": "user", "content": f"[Prior context]\n{prior}"},
                {"role": "assistant", "content": "Context loaded."},
            ])
        self.brain = brain
        self.last_active = time.time()

    def touch(self):
        self.last_active = time.time()


_sessions: dict[str, _Session] = {}


def _get_session(session_id: str | None, model_variant: str | None = None) -> _Session:
    sid = session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = _Session(sid, model_variant)
    _sessions[sid].touch()
    # Evict idle sessions (>2h) to save memory
    now = time.time()
    stale = [k for k, v in _sessions.items() if now - v.last_active > 7200 and k != sid]
    for k in stale:
        del _sessions[k]
    return _sessions[sid]


# ─────────────────────────────────────────────────────────────────────────────
#  Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model_variant: str | None = None  # 'nano' | 'core' | 'ultra'


class MemoryRequest(BaseModel):
    query: str
    session_id: str | None = None


class RememberRequest(BaseModel):
    fact: str
    session_id: str | None = None


class LoadSessionRequest(BaseModel):
    session_id: str
    target_session_id: str | None = None


class UltraRequest(BaseModel):
    task: str
    session_id: str | None = None
    model_variant: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
#  Session title persistence
# ─────────────────────────────────────────────────────────────────────────────

_TITLES_PATH = ORCA_HOME / "session_titles.json"


def _load_titles() -> dict[str, str]:
    try:
        return json.loads(_TITLES_PATH.read_text())
    except Exception:
        return {}


_session_titles: dict[str, str] = _load_titles()


def _save_title(sid: str, title: str) -> None:
    _session_titles[sid] = title
    try:
        _TITLES_PATH.write_text(json.dumps(_session_titles))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = WEB_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Orca UI not found</h1><p>Run: orca serve</p>")


@app.get("/api/status")
async def status():
    brain = get_brain(CONFIG.ollama.model_core)
    online = brain.is_available()
    model_name = "offline"
    if online:
        try:
            model_name = brain.name
        except Exception:
            model_name = "unknown"

    from orca.config import ORCA_HOME
    raw_dir = ORCA_HOME / "training" / "raw"
    raw_count = sum(
        sum(1 for _ in open(f))
        for f in raw_dir.glob("*.jsonl")
        if f.stat().st_size > 0
    ) if raw_dir.exists() else 0

    sessions = EpisodicMemory.list_sessions()

    return {
        "status": "online" if online else "offline",
        "model": model_name,
        "uptime_sec": round(time.time() - _START_TIME),
        "active_sessions": len(_sessions),
        "total_sessions": len(sessions),
        "training_examples": raw_count,
        "version": "1.0.0",
    }


@app.post("/api/chat")
async def chat(
    req: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
):
    if user:
        allowed, used, limit = check_quota(user.id, user.tier, "message")
        if not allowed:
            return JSONResponse(
                {"error": f"Daily limit reached ({used}/{limit}). Upgrade to Pro for unlimited messages."},
                status_code=429,
            )

    sess = _get_session(req.session_id, req.model_variant)
    mem_ctx = sess.memory.recall_context(req.message, n=3)
    enriched = f"[Relevant memory]\n{mem_ctx}\n\n{req.message}" if mem_ctx else req.message

    try:
        final, trace = await asyncio.to_thread(sess.agent.run, enriched)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if user:
        increment_usage(user.id, "message")

    sess.memory.add_turn("user", req.message)
    sess.memory.add_turn("assistant", final)
    sess.memory.commit_to_long_term(f"Q: {req.message[:200]}\nA: {final[:500]}")

    audit.log("chat", user_id=user.id if user else None,
              detail={"model": sess.model_variant, "tools": [tc.tool for tc in trace.tool_calls]})

    return {
        "response": final,
        "session_id": sess.id,
        "used_tools": [tc.tool for tc in trace.tool_calls],
        "plan": trace.plan_action,
    }


@app.post("/api/stream")
async def stream_chat(
    req: ChatRequest,
    user: User | None = Depends(get_current_user_optional),
):
    if user:
        allowed, used, limit = check_quota(user.id, user.tier, "message")
        if not allowed:
            async def _quota_err():
                yield f"data: {json.dumps({'type':'error','text':f'Daily limit reached ({used}/{limit}). Upgrade to Pro.'})}\n\n"
            return StreamingResponse(_quota_err(), media_type="text/event-stream")

    sess = _get_session(req.session_id, req.model_variant)
    mem_ctx = sess.memory.recall_context(req.message, n=3)
    enriched = f"[Relevant memory]\n{mem_ctx}\n\n{req.message}" if mem_ctx else req.message

    async def _event_stream() -> AsyncIterator[str]:
        # Send session_id first
        yield f"data: {json.dumps({'type': 'session', 'session_id': sess.id})}\n\n"

        full = ""
        tool_names: list[str] = []

        try:
            gen, trace = await asyncio.to_thread(
                lambda: sess.agent.stream(enriched)
            )
            # Send tool activity if any
            if trace.plan_action == "tools":
                yield f"data: {json.dumps({'type': 'thinking', 'text': 'using tools...'})}\n\n"

            for chunk in gen:
                full += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0)

            tool_names = [tc.tool for tc in trace.tool_calls]

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        # Persist
        sess.memory.add_turn("user", req.message)
        sess.memory.add_turn("assistant", full)
        sess.memory.commit_to_long_term(f"Q: {req.message[:200]}\nA: {full[:500]}")
        if user:
            increment_usage(user.id, "message")

        audit.log("stream_chat", user_id=user.id if user else None,
                  detail={"model": sess.model_variant, "tools": tool_names})

        yield f"data: {json.dumps({'type': 'done', 'tools': tool_names})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/memory/recall")
async def recall_memory(req: MemoryRequest):
    sess = _get_session(req.session_id)
    hits = sess.memory.long.recall(req.query, n=8)
    prior = sess.memory.load_prior_context()
    return {
        "hits": hits,
        "prior_context": prior[:1000] if prior else "",
        "session_id": sess.id,
    }


@app.post("/api/remember")
async def remember(req: RememberRequest):
    sess = _get_session(req.session_id)
    sess.memory.commit_to_long_term(req.fact, {"type": "user_fact", "manual": True})
    existing = sess.memory.semantic.recall_fact("all_sessions_summary") or ""
    sess.memory.semantic.store_fact(
        "all_sessions_summary",
        f"{existing}\n[User note] {req.fact}".strip()[-4000:]
    )
    return {"stored": True, "fact": req.fact, "session_id": sess.id}


@app.get("/api/sessions")
async def list_sessions():
    sessions = EpisodicMemory.list_sessions()
    return {
        "sessions": [
            {"id": s, "title": _session_titles.get(s, "")}
            for s in sessions[:50]
        ]
    }


@app.post("/api/session/load")
async def load_session(req: LoadSessionRequest):
    sess = _get_session(req.target_session_id)
    loaded = sess.memory.load_session(req.session_id)
    if loaded:
        sess.agent.load_history(sess.memory.messages())
    return {"loaded": loaded, "session_id": sess.id}


class TitleRequest(BaseModel):
    title: str


@app.patch("/api/session/{session_id}/title")
async def set_session_title(session_id: str, req: TitleRequest):
    title = req.title.strip()[:80]
    _save_title(session_id, title)
    return {"ok": True, "title": title}


@app.get("/api/session/{session_id}/export")
async def export_session(session_id: str):
    sess = _get_session(session_id)
    msgs = sess.memory.messages() if hasattr(sess.memory, "messages") else []
    title = _session_titles.get(session_id, f"Session {session_id[:8].upper()}")
    lines = [f"# {title}", f"\n_Exported from Atheris — {time.strftime('%Y-%m-%d %H:%M')}_\n", "---\n"]
    for m in msgs:
        role = "**You**" if m.get("role") == "user" else "**Atheris**"
        lines.append(f"{role}\n\n{m.get('content', '')}\n\n---\n")
    md = "\n".join(lines)
    return PlainTextResponse(
        md,
        headers={"Content-Disposition": f'attachment; filename="atheris-{session_id[:8]}.md"'},
    )


@app.post("/api/session/save")
async def save_session(req: ChatRequest):
    sid = req.session_id
    if sid and sid in _sessions:
        sess = _sessions[sid]
        sess.memory.save_session()
        return {"saved": True, "session_id": sid}
    return {"saved": False}


@app.get("/api/models")
async def list_models():
    """Return which Atheris model variants are available in Ollama."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{CONFIG.ollama.host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        pulled = {m["name"] for m in data.get("models", [])}
    except Exception:
        pulled = set()

    def _available(model_name: str) -> bool:
        if not pulled:
            return False
        base = model_name.split(":")[0]
        return model_name in pulled or any(
            m == model_name or m.split(":")[0] == base for m in pulled
        )

    return {
        "nano":  {"model": CONFIG.ollama.model_nano,  "available": _available(CONFIG.ollama.model_nano)},
        "core":  {"model": CONFIG.ollama.model_core,  "available": _available(CONFIG.ollama.model_core)},
        "ultra": {"model": CONFIG.ollama.model_ultra, "available": _available(CONFIG.ollama.model_ultra)},
    }


@app.get("/api/license")
async def license_status():
    """Return current license tier and feature set for the web UI."""
    lk = get_active_license()
    if lk:
        return {
            "tier":         lk.tier,
            "seats":        lk.seats,
            "expiry":       format_expiry(lk),
            "valid":        True,
            "has_ultra":    lk.has_feature("ultra"),
            "has_cloud":    lk.has_feature("cloud_train"),
        }
    return {
        "tier":         "free",
        "seats":        0,
        "expiry":       None,
        "valid":        False,
        "has_ultra":    False,
        "has_cloud":    False,
    }


@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Stripe webhook endpoint for automatic license fulfillment.
    Set Stripe webhook URL to: https://yourdomain.com/webhook/stripe
    """
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        from orca.license.stripe_hook import handle_stripe_event
        result = handle_stripe_event(payload, sig_header)
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": "internal error"}, status_code=500)


@app.post("/api/ultra")
async def ultra_run(req: UltraRequest):
    """SSE endpoint — runs OrcaUltra multi-agent pipeline and streams progress."""
    if not has_feature("ultra"):
        async def _gate_stream():
            yield f"data: {json.dumps({'type': 'error', 'text': 'Ultra requires a Pro license. Run: orca activate <key>'})}\n\n"
        return StreamingResponse(_gate_stream(), media_type="text/event-stream")

    sess = _get_session(req.session_id)
    progress_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def on_progress(msg: str):
        # Called from within async coroutine context (main event loop thread)
        try:
            progress_queue.put_nowait({"type": "progress", "text": msg.strip()})
        except asyncio.QueueFull:
            pass

    async def _event_stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'session', 'session_id': sess.id})}\n\n"
        yield f"data: {json.dumps({'type': 'pod_launch'})}\n\n"

        ultra = OrcaUltra(on_progress=on_progress, use_tools=True)
        pipeline_task = asyncio.create_task(ultra._run_async(req.task, max_retries=1))

        # Stream progress events while pipeline runs
        while not pipeline_task.done():
            try:
                ev = await asyncio.wait_for(progress_queue.get(), timeout=0.15)
                yield f"data: {json.dumps(ev)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        # Drain any remaining progress messages
        while not progress_queue.empty():
            try:
                ev = progress_queue.get_nowait()
                yield f"data: {json.dumps(ev)}\n\n"
            except asyncio.QueueEmpty:
                break

        try:
            pipeline = pipeline_task.result()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        # Stream final output in small chunks for smooth rendering
        text = pipeline.final_output
        chunk_size = 25
        for i in range(0, len(text), chunk_size):
            yield f"data: {json.dumps({'type': 'chunk', 'text': text[i:i+chunk_size]})}\n\n"
            await asyncio.sleep(0)

        # Persist to memory
        sess.memory.add_turn("user", req.task)
        sess.memory.add_turn("assistant", pipeline.final_output)
        sess.memory.commit_to_long_term(
            f"[Ultra] Q: {req.task[:200]}\nA: {pipeline.final_output[:500]}"
        )

        yield f"data: {json.dumps({'type': 'done', 'grade': pipeline.grade, 'iterations': pipeline.iterations})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Admin endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/audit")
async def admin_audit(
    limit: int = 100,
    admin: User = Depends(require_permission("audit_read")),
):
    return {"logs": audit.recent(limit=limit)}


@app.get("/api/admin/stats")
async def admin_stats(admin: User = Depends(require_permission("manage_users"))):
    from orca.auth.store import list_users
    users = list_users(limit=1000)
    tiers = {}
    for u in users:
        tiers[u["tier"]] = tiers.get(u["tier"], 0) + 1
    return {
        "total_users": len(users),
        "tiers": tiers,
        "active_sessions": len(_sessions),
        "uptime_sec": round(time.time() - _START_TIME),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Static files (must be last — catches everything under /static)
# ─────────────────────────────────────────────────────────────────────────────

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def create_app() -> FastAPI:
    return app
