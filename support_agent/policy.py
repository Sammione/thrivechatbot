"""Access rules applied by the mediator. Loaded from config/policy.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch

from .settings import load_yaml


@dataclass
class Policy:
    default_roles: list = field(default_factory=lambda: ["buyer", "vendor", "admin"])
    write_requires_confirmation: bool = True
    redact_fields: set = field(default_factory=set)
    max_response_chars: int = 6000
    hidden: list = field(default_factory=list)
    operations: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        d = load_yaml(path)
        return cls(
            default_roles=d.get("default_roles", ["buyer", "vendor", "admin"]),
            write_requires_confirmation=d.get("write_requires_confirmation", True),
            redact_fields={f.lower() for f in d.get("redact_fields", [])},
            max_response_chars=d.get("max_response_chars", 6000),
            hidden=d.get("hidden", []),
            operations=d.get("operations", {}) or {},
        )

    def rule(self, op) -> dict:
        if op.name in self.operations:
            return self.operations[op.name] or {}
        for pattern, r in self.operations.items():
            if fnmatch(op.name, pattern):
                return r or {}
        return {}

    def is_hidden(self, op) -> bool:
        return bool(self.rule(op).get("hidden")) or any(fnmatch(op.name, p) for p in self.hidden)

    def allowed(self, op, role) -> bool:
        return not self.is_hidden(op) and role in self.rule(op).get("roles", self.default_roles)

    def mode(self, op) -> str:
        return self.rule(op).get("mode") or ("read" if op.method == "GET" else "write")

    def binds(self, op, role) -> dict:
        r = self.rule(op)
        if "bind_roles" in r and role not in r["bind_roles"]:
            return {}
        return r.get("bind", {}) or {}

    def redactions(self, op, role) -> set:
        r = self.rule(op)
        extra = set()
        if "redact_roles" not in r or role in r["redact_roles"]:
            extra = {f.lower() for f in r.get("redact", [])}
        return self.redact_fields | extra

    def description(self, op) -> str:
        return self.rule(op).get("description", "")
