"""
The mediator sits between the LLM and every backend API. The LLM never calls a service directly.

For each call it:
  1. checks the operation exists and the user's role may use it
  2. forces identity parameters from the session (e.g. customer_id = logged-in user)
     so the AI can't read another user's data, whatever it is asked
  3. holds write actions until the user confirms them
  4. adds service credentials / forwards the user's token
  5. redacts sensitive fields from responses and writes an audit log
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx

from .policy import Policy
from .spec_loader import Operation, load_source, parse_openapi, parse_postman

PENDING_TTL = 15 * 60


@dataclass
class UserContext:
    user_id: str
    role: str
    token: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def get(self, key):
        if key == "user_id":
            return self.user_id
        if key == "role":
            return self.role
        return self.extra.get(key)


def redact(data, keys: set):
    if not keys:
        return data
    if isinstance(data, dict):
        return {k: ("[redacted]" if k.lower() in keys else redact(v, keys)) for k, v in data.items()}
    if isinstance(data, list):
        return [redact(v, keys) for v in data]
    return data


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower().replace("_", " ")))


class Mediator:
    def __init__(self, services: list, policy: Policy, http: Optional[httpx.Client] = None,
                 audit_path: Optional[str] = None):
        self.services = {s["name"]: s for s in services}
        self.policy = policy
        self.http = http or httpx.Client(timeout=20)
        self.audit_path = Path(audit_path) if audit_path else None
        self.ops: dict = {}
        self.skipped: list = []
        self.load_errors: dict = {}
        self.pending: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ loading
    def load(self):
        ops, skipped, errors = {}, [], {}
        for name, svc in self.services.items():
            try:
                if svc.get("openapi"):
                    parsed, sk = parse_openapi(load_source(svc["openapi"]), name)
                elif svc.get("postman"):
                    parsed, sk = parse_postman(load_source(svc["postman"]), name)
                else:
                    raise ValueError("service needs 'openapi' or 'postman'")
                ops.update({op.name: op for op in parsed})
                skipped += sk
            except Exception as e:
                errors[name] = f"{type(e).__name__}: {e}"
        with self._lock:
            self.ops, self.skipped, self.load_errors = ops, skipped, errors
        return len(ops)

    # ------------------------------------------------------------ discovery
    def visible_ops(self, ctx: UserContext):
        return [op for op in self.ops.values() if self.policy.allowed(op, ctx.role)]

    def tool_schema(self, op: Operation, ctx: UserContext) -> dict:
        schema = copy.deepcopy(op.input_schema)
        for param in self.policy.binds(op, ctx.role):
            if param.startswith("body."):
                body = schema["properties"].get("body", {})
                body.get("properties", {}).pop(param[5:], None)
                if param[5:] in body.get("required", []):
                    body["required"].remove(param[5:])
            else:
                schema["properties"].pop(param, None)
                if param in schema["required"]:
                    schema["required"].remove(param)
        return schema

    def describe(self, op: Operation) -> str:
        mode = self.policy.mode(op)
        svc_desc = self.services[op.service].get("description", "")
        text = f"[{op.service}{': ' + svc_desc if svc_desc else ''}] {op.method} {op.path} ({mode}). {op.summary}"
        extra = self.policy.description(op)
        if extra:
            text += f"\n{extra}"
        if mode == "write":
            text += "\nWrite action: the user will be asked to confirm before it runs."
        return text[:1000]

    def tools(self, ctx: UserContext) -> list:
        return [{"name": op.name, "description": self.describe(op), "input_schema": self.tool_schema(op, ctx)}
                for op in self.visible_ops(ctx)]

    def search(self, query: str, ctx: UserContext, k: int = 8) -> list:
        q = _tokens(query)
        scored = []
        for op in self.visible_ops(ctx):
            hay = _tokens(f"{op.name} {op.summary} {op.path} {self.policy.description(op)}")
            s = len(q & hay)
            if s:
                scored.append((s, op))
        scored.sort(key=lambda x: -x[0])
        return [{"name": op.name, "description": self.describe(op), "input_schema": self.tool_schema(op, ctx)}
                for _, op in scored[:k]]

    # ------------------------------------------------------------ calling
    def call(self, name: str, args: dict, ctx: UserContext) -> dict:
        op = self.ops.get(name)
        if not op or self.policy.is_hidden(op):
            return {"error": f"Unknown operation '{name}'"}
        if not self.policy.allowed(op, ctx.role):
            self._audit(ctx, op, args, "denied")
            return {"error": "This operation is not available for this user."}
        args = copy.deepcopy(args or {})
        for param, source in self.policy.binds(op, ctx.role).items():
            value = ctx.get(source)
            if value is None:
                return {"error": f"Missing session value '{source}' required by this operation."}
            if param.startswith("body."):
                args.setdefault("body", {})[param[5:]] = value
            else:
                args[param] = value

        if self.policy.mode(op) == "write" and self.policy.write_requires_confirmation:
            aid = uuid.uuid4().hex[:10]
            with self._lock:
                self._expire_pending()
                self.pending[aid] = {"op": name, "args": args, "user_id": ctx.user_id, "created": time.time()}
            self._audit(ctx, op, args, "pending_confirmation")
            return {"status": "pending_confirmation", "action_id": aid, "summary": self._summary(op, args),
                    "message": "NOT executed yet. Explain the action to the user and ask them to confirm."}
        return self._execute(op, args, ctx)

    def confirm(self, action_id: str, ctx: UserContext, approve: bool) -> dict:
        with self._lock:
            self._expire_pending()
            act = self.pending.get(action_id)
            if not act or act["user_id"] != ctx.user_id:
                return {"error": "No pending action with that id (it may have expired)."}
            del self.pending[action_id]
        op = self.ops.get(act["op"])
        if not op or not self.policy.allowed(op, ctx.role):
            return {"error": "This operation is no longer available."}
        if not approve:
            self._audit(ctx, op, act["args"], "cancelled")
            return {"status": "cancelled", "summary": self._summary(op, act["args"])}
        return self._execute(op, act["args"], ctx)

    def _execute(self, op: Operation, args: dict, ctx: UserContext) -> dict:
        svc = self.services[op.service]
        path = op.path
        for p in op.path_params:
            if args.get(p) in (None, ""):
                return {"error": f"Missing required parameter '{p}'"}
            path = path.replace("{" + p + "}", quote(str(args[p]), safe=""))
        params = {q: args[q] for q in op.query_params if args.get(q) is not None}
        kw = {}
        if "body" in args and op.body_mode == "json":
            kw["json"] = args["body"]
        elif "body" in args and op.body_mode == "form":
            kw["data"] = args["body"]
        try:
            r = self.http.request(op.method, svc["base_url"].rstrip("/") + path, params=params,
                                  headers=self._headers(svc, ctx), **kw)
        except httpx.HTTPError as e:
            self._audit(ctx, op, args, f"error:{type(e).__name__}")
            return {"error": f"The {op.service} service is unavailable right now."}
        try:
            data = r.json()
        except ValueError:
            data = r.text[:2000]
        result = {"status_code": r.status_code, "ok": r.is_success,
                  "data": redact(data, self.policy.redactions(op, ctx.role))}
        self._audit(ctx, op, args, r.status_code)
        return result

    # ------------------------------------------------------------ helpers
    def _headers(self, svc, ctx):
        h = dict(svc.get("headers") or {})
        auth = svc.get("auth") or {}
        import os
        if auth.get("type") == "bearer" and os.getenv(auth.get("token_env", "")):
            h["Authorization"] = f"Bearer {os.getenv(auth['token_env'])}"
        elif auth.get("type") == "header" and os.getenv(auth.get("value_env", "")):
            h[auth["name"]] = os.getenv(auth["value_env"])
        if svc.get("forward_user_token") and ctx.token:
            h["Authorization"] = f"Bearer {ctx.token}"
        return h

    def _summary(self, op, args):
        shown = {k: v for k, v in args.items() if k != "body"}
        body = args.get("body")
        txt = f"{op.summary.split(' - ')[0]} ({op.method} {op.path})"
        if shown:
            txt += f" with {json.dumps(shown, default=str)}"
        if body:
            txt += f", data {json.dumps(body, default=str)[:300]}"
        return txt

    def _expire_pending(self):
        now = time.time()
        for k in [k for k, v in self.pending.items() if now - v["created"] > PENDING_TTL]:
            del self.pending[k]

    def _audit(self, ctx, op, args, outcome):
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": ctx.user_id, "role": ctx.role, "op": op.name,
                "args": redact(args, self.policy.redact_fields), "outcome": outcome}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
