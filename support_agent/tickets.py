"""Human handoff tickets (JSONL file). Replace with your helpdesk API (Zendesk, Freshdesk...) in production."""
import json
import threading
import time
import uuid
from pathlib import Path


class TicketStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self, ctx, session_id, summary, priority="normal"):
        t = {"id": "TCK-" + uuid.uuid4().hex[:8].upper(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "user_id": ctx.user_id, "role": ctx.role, "session_id": session_id,
             "priority": priority, "summary": summary, "status": "open"}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(t) + "\n")
        return t
