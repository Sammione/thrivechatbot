"""
Turn API descriptions into a uniform list of Operations.
Supports OpenAPI 3.x (JSON/YAML, URL or file) and Postman collections v2.x.
Operations that need file uploads are skipped (the agent can't attach files).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlsplit

import httpx
import yaml

DROP_KEYS = {"title", "example", "examples", "xml", "discriminator", "externalDocs"}


@dataclass
class Operation:
    service: str
    op_id: str
    method: str
    path: str
    summary: str
    path_params: list
    query_params: list
    body_mode: Optional[str]     # json | form | None
    input_schema: dict

    @property
    def name(self) -> str:
        return f"{self.service}__{self.op_id}"[:64]


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s or "").strip("_").lower() or "op"


def load_source(src):
    if isinstance(src, dict):
        return src
    src = str(src)
    if src.startswith(("http://", "https://")):
        r = httpx.get(src, timeout=15)
        r.raise_for_status()
        text = r.text
    else:
        text = Path(src).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError:
        return yaml.safe_load(text)


# ------------------------------------------------------------------ OpenAPI
def _deref(spec, ref):
    node = spec
    for part in ref.lstrip("#/").split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _clean(schema, spec, depth=0):
    if depth > 12:
        return {}
    if isinstance(schema, list):
        return [_clean(x, spec, depth + 1) for x in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return _clean(_deref(spec, schema["$ref"]), spec, depth + 1)
    out = {}
    for k, v in schema.items():
        if k in DROP_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pn: _clean(pv, spec, depth + 1) for pn, pv in v.items()}
        else:
            out[k] = _clean(v, spec, depth + 1)
    return out


def _has_binary(schema) -> bool:
    if isinstance(schema, dict):
        if schema.get("format") == "binary" or "contentMediaType" in schema:
            return True
        return any(_has_binary(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_binary(v) for v in schema)
    return False


def parse_openapi(spec: dict, service: str):
    ops, skipped = [], []
    for path, item in (spec.get("paths") or {}).items():
        common = item.get("parameters", [])
        for method in ("get", "post", "put", "patch", "delete"):
            op = item.get(method)
            if not op:
                continue
            op_id = _slug(op.get("operationId") or f"{method}_{path}")
            props, required, pp, qp = {}, [], [], []
            for prm in common + op.get("parameters", []):
                prm = _clean(prm, spec)
                loc = prm.get("in")
                if loc not in ("path", "query"):
                    continue
                name = prm["name"]
                sch = copy.deepcopy(prm.get("schema") or {"type": "string"})
                if prm.get("description"):
                    sch["description"] = prm["description"]
                props[name] = sch
                (pp if loc == "path" else qp).append(name)
                if loc == "path" or prm.get("required"):
                    required.append(name)

            body_mode = None
            if op.get("requestBody"):
                rb = _clean(op["requestBody"], spec)
                content = rb.get("content", {})
                if "application/json" in content:
                    body_mode, bs = "json", content["application/json"].get("schema", {})
                elif "application/x-www-form-urlencoded" in content:
                    body_mode, bs = "form", content["application/x-www-form-urlencoded"].get("schema", {})
                elif "multipart/form-data" in content and not _has_binary(content["multipart/form-data"]):
                    body_mode, bs = "form", content["multipart/form-data"].get("schema", {})
                else:
                    skipped.append(f"{service}__{op_id} (file upload / unsupported body)")
                    continue
                props["body"] = bs
                if rb.get("required"):
                    required.append("body")

            summary = " - ".join(x for x in (op.get("summary"), op.get("description")) if x) or f"{method.upper()} {path}"
            ops.append(Operation(service, op_id, method.upper(), path, summary.strip(), pp, qp, body_mode,
                                 {"type": "object", "properties": props, "required": required}))
    return _dedupe(ops), skipped


# ------------------------------------------------------------------ Postman
def _schema_from_example(v):
    if isinstance(v, dict):
        return {"type": "object", "properties": {k: _schema_from_example(x) for k, x in v.items()}}
    if isinstance(v, list):
        return {"type": "array", "items": _schema_from_example(v[0]) if v else {}}
    if isinstance(v, bool):
        return {"type": "boolean"}
    if isinstance(v, int):
        return {"type": "integer"}
    if isinstance(v, float):
        return {"type": "number"}
    if isinstance(v, str):
        return {"type": "string"}
    return {}


def _parse_raw_url(raw: str) -> dict:
    raw = re.sub(r"^\{\{\w+\}\}", "", raw.strip())
    if not raw.startswith("http"):
        raw = "http://host" + (raw if raw.startswith("/") else "/" + raw)
    u = urlsplit(raw)
    return {"path": u.path.split("/"), "query": [{"key": k, "value": v} for k, v in parse_qsl(u.query)]}


def _text(d):
    if isinstance(d, dict):
        return d.get("content", "")
    return d or ""


def parse_postman(coll: dict, service: str):
    ops, skipped = [], []

    def walk(items, prefix):
        for it in items:
            if "item" in it:
                walk(it["item"], prefix + [it.get("name", "")])
                continue
            req = it.get("request")
            if not req:
                continue
            if isinstance(req, str):
                req = {"method": "GET", "url": req}
            method = (req.get("method") or "GET").upper()
            url = req.get("url") or ""
            url = _parse_raw_url(url) if isinstance(url, str) else url
            segs_in = url.get("path") or []
            if isinstance(segs_in, str):
                segs_in = segs_in.split("/")

            props, required, pp, qp, segs = {}, [], [], [], []
            for seg in segs_in:
                if not seg:
                    continue
                m = re.fullmatch(r":(\w+)", seg) or re.fullmatch(r"\{\{(\w+)\}\}", seg)
                if m:
                    n = m.group(1)
                    segs.append("{" + n + "}"); pp.append(n); required.append(n)
                    props[n] = {"type": "string"}
                else:
                    segs.append(seg)
            for v in url.get("variable") or []:
                if v.get("key") in props and v.get("description"):
                    props[v["key"]]["description"] = _text(v["description"])
            for q in url.get("query") or []:
                if q.get("disabled") or not q.get("key"):
                    continue
                props[q["key"]] = {"type": "string"}
                if q.get("description"):
                    props[q["key"]]["description"] = _text(q["description"])
                qp.append(q["key"])

            body, body_mode = req.get("body") or {}, None
            mode = body.get("mode")
            if mode == "raw":
                try:
                    example = json.loads(body.get("raw") or "")
                except ValueError:
                    example = None
                if isinstance(example, dict):
                    props["body"], body_mode = _schema_from_example(example), "json"
            elif mode in ("urlencoded", "formdata"):
                fields = [f for f in body.get(mode, []) if not f.get("disabled")]
                if any(f.get("type") == "file" for f in fields):
                    skipped.append(f"{service}__{it.get('name')} (file upload)")
                    continue
                props["body"] = {"type": "object", "properties": {f["key"]: {"type": "string"} for f in fields}}
                body_mode = "form"

            path = "/" + "/".join(segs)
            op_id = _slug("_".join([p for p in prefix if p] + [it.get("name") or f"{method}_{path}"]))
            summary = _text(req.get("description")) or it.get("name") or f"{method} {path}"
            ops.append(Operation(service, op_id, method, path, summary.strip(), pp, qp, body_mode,
                                 {"type": "object", "properties": props, "required": required}))

    walk(coll.get("item", []), [])
    return _dedupe(ops), skipped


def _dedupe(ops):
    seen = {}
    for op in ops:
        base, n = op.op_id, 2
        while op.name in seen:
            op.op_id = f"{base}_{n}"; n += 1
        seen[op.name] = op
    return list(seen.values())
