"""
Support agent API. Run:  uvicorn support_agent.server:app --port 8001

Your backend (which already logged the user in) calls this with the user's identity in headers:
  X-User-Id, X-User-Role (buyer|vendor|admin), optional X-User-Token (forwarded to services
  with forward_user_token: true), and X-Agent-Key if AGENT_API_KEY is set.
Never let browsers set these headers directly in production.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import SessionStore, SupportAgent
from .knowledge import KnowledgeBase
from .mediator import Mediator, UserContext
from .policy import Policy
from .settings import AgentSettings, load_yaml
from .tickets import TicketStore

settings = AgentSettings()
state: dict = {}


def build(settings: AgentSettings, client=None, http=None):
    policy = Policy.load(settings.policy_file)
    services = load_yaml(settings.services_file).get("services", [])
    mediator = Mediator(services, policy, http=http, audit_path=f"{settings.data_dir}/audit.jsonl")
    mediator.load()
    kb = KnowledgeBase(settings.knowledge_dir)
    tickets = TicketStore(f"{settings.data_dir}/tickets.jsonl")
    agent = SupportAgent(settings, mediator, kb, tickets, client=client)
    return {"mediator": mediator, "kb": kb, "agent": agent,
            "sessions": SessionStore(max_history=settings.max_history)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not state:
        state.update(build(settings))
    yield


app = FastAPI(title="Customer Support Agent", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


def user_ctx(x_user_id: str = Header(...), x_user_role: str = Header("buyer"),
             x_user_token: Optional[str] = Header(None), x_agent_key: Optional[str] = Header(None)):
    if settings.api_key and x_agent_key != settings.api_key:
        raise HTTPException(401, "Invalid agent key")
    if x_user_role not in settings.roles:
        raise HTTPException(400, f"Unknown role '{x_user_role}'")
    return UserContext(x_user_id, x_user_role, x_user_token)


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class ConfirmIn(BaseModel):
    session_id: str
    action_id: str
    approve: bool


def _session(session_id, ctx):
    try:
        return state["sessions"].get_or_create(session_id, ctx)
    except PermissionError as e:
        raise HTTPException(403, str(e))


def _run(session, ctx):
    if not state["mediator"].ops:      # services may have been down at startup
        state["mediator"].load()
    try:
        out = state["agent"].run(session, ctx)
    except Exception as e:
        raise HTTPException(502, f"AI service error: {type(e).__name__}")
    state["sessions"].trim(session)
    return {"session_id": session.id, **out}


@app.post("/chat")
def chat(body: ChatIn, ctx: UserContext = Depends(user_ctx)):
    """Send a user message; returns the reply, and pending_action if something needs confirming."""
    session = _session(body.session_id, ctx)
    session.messages.append({"role": "user", "content": body.message})
    return _run(session, ctx)


@app.post("/chat/confirm")
def confirm(body: ConfirmIn, ctx: UserContext = Depends(user_ctx)):
    """User pressed Confirm/Cancel on a pending action."""
    session = state["sessions"].get(body.session_id)
    if not session or session.user_id != ctx.user_id:
        raise HTTPException(404, "Session not found")
    result = state["mediator"].confirm(body.action_id, ctx, body.approve)
    verdict = "approved" if body.approve else "declined"
    session.messages.append({"role": "user", "content":
        f"[System notice: the user {verdict} pending action {body.action_id}. "
        f"Result: {json.dumps(result, default=str)[:3000]}] Tell the user the outcome briefly."})
    return _run(session, ctx)


@app.get("/tools")
def list_tools(ctx: UserContext = Depends(user_ctx)):
    """Debug: which operations this user's role can use, plus load problems."""
    m = state["mediator"]
    return {"operations": [t["name"] for t in m.tools(ctx)], "skipped": m.skipped, "load_errors": m.load_errors}


@app.post("/admin/reload")
def reload(ctx: UserContext = Depends(user_ctx)):
    """Re-read API specs, policy and help articles (after you deploy new endpoints)."""
    if ctx.role != "admin":
        raise HTTPException(403, "Admins only")
    sessions = state.get("sessions")
    state.update(build(settings))
    if sessions:
        state["sessions"] = sessions
    return {"operations": len(state["mediator"].ops), "load_errors": state["mediator"].load_errors}


@app.get("/", response_class=HTMLResponse)
def demo_page():
    """Test chat page (development only)."""
    return (Path(__file__).parent / "static" / "chat.html").read_text(encoding="utf-8")
