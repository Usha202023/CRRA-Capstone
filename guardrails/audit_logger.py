"""
guardrails/audit_logger.py - Append-only audit trail for the CRRA agents.

Every call to AuditLogger.log() writes one JSON object per line to
logs/audit_trail.jsonl (relative to the project root) and prints it.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "audit_trail.jsonl"


class AuditLogger:
    """Writes one JSON line per event. Safe to share across nodes in one process."""

    def __init__(self, log_path: Path | str = DEFAULT_LOG_PATH, run_id: str | None = None,
                 echo: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.echo = echo
        self._lock = threading.Lock()

    def log(self, node: str, event: str, contract_id: str | None = None, **details) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "node": node,
            "event": event,
            "contract_id": contract_id,
            "details": details,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if self.echo:
            print(f"[AUDIT] {line}")
        return entry