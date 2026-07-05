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

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orca.license import current_tier, get_active_license, has_feature
from orca.license.keys import format_expiry

from orca.auth import auth_router, get_current_user, get_current_user_optional, check_quota, increment_usage
from orca.auth.rbac import require_permission
from orca.auth.store import User
from orca import audit

from orca.brain.providers import get_brain
from orca.brain.memory import MemoryEngine, EpisodicMemory
from orca.brain.agent import AgentLoop
from orca.brain.context import ContextManager
from orca.tools import build_registry
from orca.character import CORE_SYSTEM_WITH_TOOLS
from orca.personas import get_persona_system
from orca.config import CONFIG, ORCA_HOME
from orca.variants.ultra import OrcaUltra
from orca.docs import (
    extract, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE,
    chunk_text, DocStore, register_doc, unregister_doc, list_docs,
    run_deep_rag,
)
from orca.docs.citation_check import check_citations
from orca.docs.pii_redact import redact_pii
from orca.brain.explainability import ExplainStore, build_from_rag_result
from orca.brain.knowledge_graph import KnowledgeGraph
from orca.serve import session_store, ratelimit, metrics
from orca.serve.moderation import check_input, CRISIS_RESOURCES
from orca.code import run_code

_START_TIME = time.time()
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Orca API", version="1.0.0", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Records request count/status/latency per endpoint. Uses the matched
    ROUTE TEMPLATE (e.g. "/api/explain/{session_id}/{message_id}"), not the
    raw path with real session IDs substituted in — using raw paths as
    metric labels would create unbounded cardinality (a new "endpoint" for
    every distinct session/doc/message ID ever seen), which is exactly the
    kind of metrics-system footgun that quietly blows up memory in
    production. route.path is only available on request.scope AFTER
    routing has resolved, which happens by the time call_next() returns.
    """
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000

    route = request.scope.get("route")
    endpoint = route.path if route else request.url.path
    metrics.record_request(f"{request.method} {endpoint}", response.status_code, duration_ms)

    return response


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

        # Cross-instance continuity: if Redis has this session's state (from
        # this or another API instance), restore the exact conversation
        # history instead of starting cold. An explicit model_variant from
        # the caller wins over the persisted one (user actively switching
        # models mid-conversation should be respected); Redis only fills in
        # when the caller didn't specify (session_id known, variant omitted).
        redis_state = session_store.load_session_state(session_id)
        restored_history = None
        if redis_state:
            model_variant = model_variant or redis_state.get("model_variant")
            restored_history = redis_state.get("history")

        self.model_variant = model_variant or "core"
        self.memory = MemoryEngine(session_id=session_id)
        brain = get_brain(_model_name_for_variant(model_variant))
        self.ctx = ContextManager(brain)
        tools = build_registry(memory_engine=self.memory)
        self.agent = AgentLoop(brain=brain, tools=tools, session_id=session_id)

        if restored_history:
            self.agent.load_history(restored_history)
        else:
            # Fallback: no exact turn history available (fresh session, or
            # Redis disabled/empty) — reconstruct rough context from the
            # long-term summary store instead. Less precise than Redis's
            # exact history, only used when that's unavailable.
            prior = self.memory.load_prior_context()
            if prior:
                self.agent.load_history([
                    {"role": "user", "content": f"[Prior context]\n{prior}"},
                    {"role": "assistant", "content": "Context loaded."},
                ])

        self.brain = brain
        self.doc_store = DocStore(session_id=session_id, ollama_host=CONFIG.ollama.host)
        self.explain_store = ExplainStore()
        self.knowledge_graph = KnowledgeGraph(session_id=session_id)
        self.last_active = time.time()

    def touch(self):
        self.last_active = time.time()

    def persist_to_redis(self):
        """Save current conversation history so any instance can pick this session back up."""
        session_store.save_session_state(self.id, self.model_variant, self.agent.get_history())


_sessions: dict[str, _Session] = {}


def _get_session(session_id: str | None, model_variant: str | None = None, user_id: str | None = None) -> _Session:
    sid = session_id or str(uuid.uuid4())
    if sid not in _sessions:
        _sessions[sid] = _Session(sid, model_variant)
    _sessions[sid].touch()
    session_store.touch_session(sid)  # refresh Redis TTL — no-op if Redis disabled

    # Records which account this session belongs to — without this, "delete
    # my account" has no way to find and remove the chat history/documents
    # tied to it, since sessions are otherwise anonymous by session_id alone.
    if user_id:
        from orca.auth.store import record_user_session
        record_user_session(user_id, sid)

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
        "redis": {"enabled": session_store.enabled(), "reachable": session_store.ping()},
    }


@app.post("/api/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    # IP-based floor applies regardless of auth status — the per-user tier
    # quota below only ever ran `if user:`, leaving anonymous requests with
    # zero limit. This closes that gap; authenticated users still also get
    # their tier quota checked right after.
    ratelimit.enforce(request, ratelimit.CHAT_ANY, extra_key="chat")

    if user:
        allowed, used, limit = check_quota(user.id, user.tier, "message")
        if not allowed:
            return JSONResponse(
                {"error": f"Daily limit reached ({used}/{limit}). Upgrade to Pro for unlimited messages."},
                status_code=429,
            )

    # Input moderation — checked before the message ever reaches the model.
    # BLOCK: hard refusal, generation never happens. SUPPORT (self-harm):
    # never blocked — crisis resources get injected into context instead,
    # since refusing someone in crisis is the opposite of good practice.
    # FLAG: logged for visibility, generation proceeds unchanged.
    mod_result = check_input(req.message)
    if mod_result.action == "block":
        audit.log("input_moderation_blocked", user_id=user.id if user else None,
                  detail={"categories": mod_result.flagged_categories})
        return JSONResponse(
            {"error": "This request can't be processed — it matches a category we don't generate content for."},
            status_code=400,
        )
    if mod_result.action in ("support", "flag"):
        audit.log(f"input_moderation_{mod_result.action}", user_id=user.id if user else None,
                  detail={"categories": mod_result.flagged_categories})

    sess = _get_session(req.session_id, req.model_variant, user_id=user.id if user else None)
    mem_ctx = sess.memory.recall_context(req.message, n=3)
    enriched = f"[Relevant memory]\n{mem_ctx}\n\n{req.message}" if mem_ctx else req.message
    persona_system = get_persona_system(sess.model_variant)
    if mod_result.action == "support":
        persona_system += (
            f"\n\nIMPORTANT: This message may indicate the user is in emotional distress or crisis. "
            f"Respond with warmth and care. Include these resources naturally in your response:\n{CRISIS_RESOURCES}"
        )

    try:
        final, trace = await asyncio.to_thread(sess.agent.run, enriched, persona_system)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if user:
        increment_usage(user.id, "message")

    sess.memory.add_turn("user", req.message)
    sess.memory.add_turn("assistant", final)
    sess.memory.commit_to_long_term(f"Q: {req.message[:200]}\nA: {final[:500]}")
    sess.persist_to_redis()

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
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    ratelimit.enforce(request, ratelimit.CHAT_ANY, extra_key="stream")

    if user:
        allowed, used, limit = check_quota(user.id, user.tier, "message")
        if not allowed:
            async def _quota_err():
                yield f"data: {json.dumps({'type':'error','text':f'Daily limit reached ({used}/{limit}). Upgrade to Pro.'})}\n\n"
            return StreamingResponse(_quota_err(), media_type="text/event-stream")

    mod_result = check_input(req.message)
    if mod_result.action == "block":
        audit.log("input_moderation_blocked", user_id=user.id if user else None,
                  detail={"categories": mod_result.flagged_categories})
        _mod_block_msg = "This request can't be processed — it matches a category we don't generate content for."
        async def _mod_block():
            yield f"data: {json.dumps({'type': 'error', 'text': _mod_block_msg})}\n\n"
        return StreamingResponse(_mod_block(), media_type="text/event-stream")
    if mod_result.action in ("support", "flag"):
        audit.log(f"input_moderation_{mod_result.action}", user_id=user.id if user else None,
                  detail={"categories": mod_result.flagged_categories})

    sess = _get_session(req.session_id, req.model_variant, user_id=user.id if user else None)
    mem_ctx = sess.memory.recall_context(req.message, n=3)
    enriched = f"[Relevant memory]\n{mem_ctx}\n\n{req.message}" if mem_ctx else req.message

    # Deep RAG: 7-stage pipeline (query intelligence → multi-signal recall →
    # RRF fusion → rerank → sufficiency check → citation DNA). Only runs if
    # docs are loaded for this session.
    rag_result = None
    if sess.doc_store.count() > 0:
        history = sess.memory.messages() if hasattr(sess.memory, "messages") else []
        history_strs = [f"{m.get('role','')}: {m.get('content','')[:200]}" for m in history[-6:]]
        rag_result = await asyncio.to_thread(
            run_deep_rag,
            req.message,
            sess.doc_store,
            history_strs,
            CONFIG.ollama.host,
            _model_name_for_variant(req.model_variant),
        )
        if rag_result.context_block:
            enriched = f"[Document context — cite sources as [D1], [D2], etc.]\n{rag_result.context_block}\n\n{enriched}"

    persona_system = get_persona_system(sess.model_variant)
    if mod_result.action == "support":
        persona_system += (
            f"\n\nIMPORTANT: This message may indicate the user is in emotional distress or crisis. "
            f"Respond with warmth and care. Include these resources naturally in your response:\n{CRISIS_RESOURCES}"
        )
    message_id = str(uuid.uuid4())

    async def _event_stream() -> AsyncIterator[str]:
        # Send session_id first
        yield f"data: {json.dumps({'type': 'session', 'session_id': sess.id})}\n\n"
        if rag_result and rag_result.sources:
            yield f"data: {json.dumps({'type': 'rag', 'sources': rag_result.sources, 'confidence': rag_result.sufficiency_confidence, 'rounds': rag_result.retrieval_rounds, 'contradictions': rag_result.contradictions})}\n\n"

        full = ""
        tool_names: list[str] = []
        plan_action = "direct"

        try:
            gen, trace = await asyncio.to_thread(
                lambda: sess.agent.stream(enriched, persona_system)
            )
            # Send tool activity if any
            if trace.plan_action == "tools":
                yield f"data: {json.dumps({'type': 'thinking', 'text': 'using tools...'})}\n\n"

            for chunk in gen:
                full += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0)

            tool_names = [tc.tool for tc in trace.tool_calls]
            plan_action = trace.plan_action

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            return

        # Persist
        sess.memory.add_turn("user", req.message)
        sess.memory.add_turn("assistant", full)
        sess.memory.commit_to_long_term(f"Q: {req.message[:200]}\nA: {full[:500]}")
        sess.persist_to_redis()
        if user:
            increment_usage(user.id, "message")

        # Knowledge graph extraction — fire-and-forget, NOT awaited. This is
        # an LLM call (real latency on CPU inference per this project's own
        # benchmarks), and the user is already looking at their finished
        # response by this point — making them wait for graph extraction
        # too would be pure UX cost for a background-enrichment feature.
        # Failures here are silent by design (extract_and_add never raises)
        # and don't affect the conversation that already completed.
        asyncio.create_task(asyncio.to_thread(
            sess.knowledge_graph.extract_and_add,
            f"{req.message}\n{full}", "chat", sess.brain,
        ))

        # Citation compliance: mechanical check, not just a prompt instruction.
        # If document context was available and the response cited zero
        # sources, that's a real governance signal — logged for visibility
        # and surfaced to the frontend so it can flag the answer, not silently
        # trusted just because the system prompt said to cite.
        context_block = rag_result.context_block if rag_result else ""
        citation_report = check_citations(full, context_block)
        if citation_report["had_context"] and not citation_report["compliant"]:
            audit.log("citation_compliance_failed", user_id=user.id if user else None,
                      detail={"message_id": message_id, "note": citation_report["note"]})

        audit.log("stream_chat", user_id=user.id if user else None,
                  detail={"model": sess.model_variant, "tools": tool_names})

        # Explainability: capture the full retrieval/reasoning trace for this
        # message, keyed by message_id so the frontend "Explain" button can
        # fetch it on demand without bloating every SSE payload.
        explain_record = build_from_rag_result(message_id, rag_result, plan_action, tool_names)
        sess.explain_store.add(explain_record)

        yield f"data: {json.dumps({'type': 'done', 'tools': tool_names, 'message_id': message_id, 'citation_compliance': citation_report})}\n\n"

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


@app.get("/api/explain/{session_id}/{message_id}")
async def explain_answer(session_id: str, message_id: str):
    """
    'Explain this answer' — full retrieval chain, query intelligence, citation
    DNA, sufficiency confidence, contradictions, and agent reasoning trace
    for a specific assistant message.
    """
    sess = _get_session(session_id)
    record = sess.explain_store.get(message_id)
    if record is None:
        return JSONResponse(
            {"error": "No explain data found for this message. It may have expired (max 50 kept per session) or belong to a different session."},
            status_code=404,
        )
    return record.to_dict()


@app.get("/api/knowledge/{session_id}")
async def knowledge_graph_summary(session_id: str):
    """Lists every entity the knowledge graph has extracted for this session."""
    sess = _get_session(session_id)
    return {
        "session_id": session_id,
        "count": sess.knowledge_graph.count(),
        "entities": sess.knowledge_graph.all_entities(),
    }


@app.get("/api/knowledge/{session_id}/{entity_name}")
async def knowledge_graph_entity(session_id: str, entity_name: str):
    """Full detail on one entity — its relationships as subject and object, plus one-hop neighbors."""
    sess = _get_session(session_id)
    info = sess.knowledge_graph.query_entity(entity_name)
    if info is None:
        return JSONResponse({"error": f"No entity '{entity_name}' found in this session's knowledge graph."}, status_code=404)
    info["neighbors"] = sess.knowledge_graph.neighbors(entity_name)
    return info


@app.get("/api/session/{session_id}/export")
async def export_session(session_id: str):
    sess = _get_session(session_id)
    msgs = sess.memory.messages() if hasattr(sess.memory, "messages") else []
    title = _session_titles.get(session_id, f"Session {session_id[:8].upper()}")
    lines = [f"# {title}", f"\n_Exported from Orca — {time.strftime('%Y-%m-%d %H:%M')}_\n", "---\n"]
    for m in msgs:
        role = "**You**" if m.get("role") == "user" else "**Orca**"
        lines.append(f"{role}\n\n{m.get('content', '')}\n\n---\n")
    md = "\n".join(lines)
    return PlainTextResponse(
        md,
        headers={"Content-Disposition": f'attachment; filename="orca-{session_id[:8]}.md"'},
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
    """Return which Orca model variants are available in Ollama."""
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


LEGAL_DIR = Path(__file__).parent.parent.parent / "legal"


@app.get("/legal/privacy", response_class=PlainTextResponse)
async def privacy_policy():
    path = LEGAL_DIR / "PRIVACY_POLICY.md"
    if not path.exists():
        return PlainTextResponse("Privacy policy not found.", status_code=404)
    return PlainTextResponse(path.read_text(), media_type="text/markdown")


@app.get("/legal/terms", response_class=PlainTextResponse)
async def terms_of_service():
    path = LEGAL_DIR / "TERMS_OF_SERVICE.md"
    if not path.exists():
        return PlainTextResponse("Terms of service not found.", status_code=404)
    return PlainTextResponse(path.read_text(), media_type="text/markdown")


@app.get("/legal/ai-policy", response_class=PlainTextResponse)
async def ai_policy():
    """AI Policy & Risk Register — the policy Orca's technical safety controls actually enforce."""
    path = LEGAL_DIR / "AI_POLICY.md"
    if not path.exists():
        return PlainTextResponse("AI policy document not found.", status_code=404)
    return PlainTextResponse(path.read_text(), media_type="text/markdown")


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


@app.post("/api/billing/checkout")
async def create_checkout(
    tier: str = "pro",
    interval: str = "month",
    user: User = Depends(get_current_user),
):
    """
    Creates a Stripe Checkout Session for the AUTHENTICATED web user and
    returns its URL for the frontend to redirect to. client_reference_id is
    set to this user's id — the webhook handler reads it back to know WHICH
    web account to upgrade once payment completes. Without this, a completed
    Stripe payment had no way to connect back to a specific logged-in user's
    account tier (see orca/license/stripe_hook.py for the other half of this).
    """
    import os as _os
    try:
        import stripe
    except ImportError:
        return JSONResponse({"error": "Stripe is not installed on this server."}, status_code=503)

    stripe_secret = _os.environ.get("STRIPE_SECRET_KEY")
    if not stripe_secret:
        return JSONResponse({"error": "Stripe is not configured on this server."}, status_code=503)
    stripe.api_key = stripe_secret

    price_env_map = {
        ("pro", "month"):        "STRIPE_PRICE_PRO",
        ("pro", "year"):         "STRIPE_PRICE_PRO_YEAR",
        ("enterprise", "month"): "STRIPE_PRICE_ENT",
        ("enterprise", "year"):  "STRIPE_PRICE_ENT_YEAR",
    }
    price_id = _os.environ.get(price_env_map.get((tier, interval), "STRIPE_PRICE_PRO"), "")
    if not price_id:
        return JSONResponse(
            {"error": f"No Stripe price configured for tier='{tier}' interval='{interval}'."},
            status_code=503,
        )

    from orca.auth.store import get_stripe_customer_id
    existing_customer_id = get_stripe_customer_id(user.id)

    base_url = _os.environ.get("ORCA_PUBLIC_URL", "http://localhost:7337")

    session_kwargs = dict(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        client_reference_id=user.id,
        metadata={"user_id": user.id, "tier": tier},
        success_url=f"{base_url}/?checkout=success",
        cancel_url=f"{base_url}/?checkout=cancelled",
    )
    # Reuse the existing Stripe Customer if this user has paid before —
    # avoids creating duplicate Customer records on every checkout attempt.
    if existing_customer_id:
        session_kwargs["customer"] = existing_customer_id
    else:
        session_kwargs["customer_email"] = user.email

    try:
        session = stripe.checkout.Session.create(**session_kwargs)
    except Exception as e:
        return JSONResponse({"error": f"Could not create checkout session: {e}"}, status_code=502)

    return {"url": session.url}


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
        # Note: OrcaUltra doesn't run through AgentLoop, so this turn won't
        # appear in agent.get_history() — persist_to_redis() still preserves
        # model_variant continuity even though the turn itself isn't captured.
        sess.persist_to_redis()

        yield f"data: {json.dumps({'type': 'done', 'grade': pipeline.grade, 'iterations': pipeline.iterations})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Document Q&A (RAG) endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/docs/upload")
async def upload_doc(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
    user: User | None = Depends(get_current_user_optional),
):
    """Upload a document for RAG — extract, chunk, embed, and store."""
    ratelimit.enforce(request, ratelimit.DOC_UPLOAD, extra_key="doc_upload")
    if file.filename and Path(file.filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
            status_code=400,
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        return JSONResponse(
            {"error": f"File too large ({len(data)//1024//1024}MB). Max: {MAX_FILE_SIZE//1024//1024}MB"},
            status_code=413,
        )

    filename = file.filename or "upload.txt"
    try:
        text = extract(filename, data)
    except Exception as e:
        return JSONResponse({"error": f"Extraction failed: {e}"}, status_code=422)

    if not text.strip():
        return JSONResponse({"error": "No text could be extracted from this file."}, status_code=422)

    # PII redaction — scrubs SSNs, emails, phone numbers, and Luhn-valid
    # credit card numbers BEFORE the document is chunked/embedded into the
    # persistent vector store. Applies here (uploads), not to live chat
    # messages — see orca/docs/pii_redact.py docstring for why that scope
    # boundary is deliberate, not an oversight.
    text, pii_report = redact_pii(text)
    if pii_report["total"] > 0:
        audit.log("doc_pii_redacted", user_id=user.id if user else None,
                  detail={"filename": filename, **{k: v for k, v in pii_report.items() if k != "total"},
                          "total_redactions": pii_report["total"]})

    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text, doc_id=doc_id, filename=filename)

    sess = _get_session(session_id, user_id=user.id if user else None)
    stored = sess.doc_store.add_chunks(chunks, doc_id=doc_id, filename=filename)
    register_doc(sess.id, doc_id, filename, chunk_count=stored, size_bytes=len(data))

    audit.log("doc_upload", user_id=user.id if user else None,
              detail={"filename": filename, "chunks": stored, "bytes": len(data),
                      "pii_redactions": pii_report["total"]})

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": stored,
        "size_bytes": len(data),
        "session_id": sess.id,
    }


@app.get("/api/docs/list")
async def list_session_docs(session_id: str | None = None):
    """List all documents uploaded in the current session."""
    sess = _get_session(session_id)
    docs = list_docs(sess.id)
    return {"docs": docs, "session_id": sess.id, "total_chunks": sess.doc_store.count()}


@app.delete("/api/docs/{doc_id}")
async def delete_doc(
    doc_id: str,
    session_id: str | None = None,
    user: User | None = Depends(get_current_user_optional),
):
    """Remove a document and all its chunks from the session store."""
    sess = _get_session(session_id, user_id=user.id if user else None)
    ok = sess.doc_store.delete_doc(doc_id)
    unregister_doc(sess.id, doc_id)
    audit.log("doc_delete", user_id=user.id if user else None, detail={"doc_id": doc_id})
    return {"deleted": ok, "doc_id": doc_id, "session_id": sess.id}


# ─────────────────────────────────────────────────────────────────────────────
#  Code Interpreter endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CodeRunRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: str | None = None


@app.post("/api/code/run")
async def code_run(
    req: CodeRunRequest,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
):
    """Execute Python code in a sandboxed subprocess. Returns stdout/stderr/error."""
    ratelimit.enforce(request, ratelimit.CODE_RUN, extra_key="code_run")
    if req.language != "python":
        return JSONResponse({"error": f"Language '{req.language}' not supported. Only Python is available."}, status_code=400)

    result = await asyncio.to_thread(run_code, req.code)

    audit.log("code_run", user_id=user.id if user else None,
              detail={"exit_code": result.exit_code, "duration_ms": result.duration_ms,
                      "ok": result.ok})

    return {
        "stdout":      result.stdout,
        "stderr":      result.stderr,
        "error":       result.error,
        "exit_code":   result.exit_code,
        "duration_ms": result.duration_ms,
        "ok":          result.ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Monitoring
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/metrics")
async def admin_metrics(admin: User = Depends(require_permission("audit_read"))):
    """JSON metrics snapshot — request counts, error rates, latency percentiles per endpoint."""
    return metrics.get_metrics_snapshot()


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus exposition format — no auth, standard scrape convention.
    If exposing this port publicly, put a firewall/reverse-proxy rule in
    front of it; this endpoint reveals operational detail (request volumes,
    error rates, endpoint names) that shouldn't be internet-visible.
    """
    return PlainTextResponse(metrics.get_prometheus_text(), media_type="text/plain; version=0.0.4")


# ─────────────────────────────────────────────────────────────────────────────
#  Admin endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/admin/audit")
async def admin_audit(
    limit: int = 100,
    admin: User = Depends(require_permission("audit_read")),
):
    return {"logs": audit.recent(limit=limit)}


@app.get("/api/admin/audit/verify")
async def admin_audit_verify(
    start_seq: int = 0,
    end_seq: int | None = None,
    admin: User = Depends(require_permission("audit_read")),
):
    """Recompute the hash chain and report any tampering — the compliance check."""
    return audit.verify_chain(start_seq=start_seq, end_seq=end_seq)


@app.get("/api/admin/audit/export")
async def admin_audit_export(
    start_seq: int = 0,
    end_seq: int | None = None,
    admin: User = Depends(require_permission("audit_read")),
):
    """Court-admissible export: full entries + verification + a top-level signature."""
    export = audit.export_for_audit(start_seq=start_seq, end_seq=end_seq)
    return JSONResponse(
        export,
        headers={"Content-Disposition": f'attachment; filename="orca-audit-export-{int(time.time())}.json"'},
    )


@app.get("/api/admin/governance/cards")
async def admin_list_model_cards(admin: User = Depends(require_permission("audit_read"))):
    """List every generated model card with a quick signature-validity check."""
    from orca.governance import list_model_cards
    return {"cards": list_model_cards()}


@app.get("/api/admin/governance/cards/{variant}")
async def admin_get_model_card(variant: str, admin: User = Depends(require_permission("audit_read"))):
    """Full model card for a variant (nano/core/ultra), including safety scores and limitations."""
    from orca.governance import load_model_card, verify_model_card
    card = load_model_card(variant)
    if card is None:
        return JSONResponse({"error": f"No model card found for variant '{variant}'. Run `orca train card {variant}` first."}, status_code=404)
    verification = verify_model_card(variant)
    return {"card": card.to_dict(), "verification": verification}


@app.post("/api/admin/governance/cards/{variant}/generate")
async def admin_generate_model_card(variant: str, admin: User = Depends(require_permission("manage_users"))):
    """Regenerate a model card from the latest eval + red-team reports on disk."""
    from orca.governance import generate_model_card
    try:
        card = generate_model_card(variant)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    audit.log("model_card_generated", user_id=admin.id, detail={"variant": variant})
    return {"card": card.to_dict()}


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
