"""
Art. 12 — Record-keeping.

Append-only, hash-chained SQLite audit log.  Every record includes
SHA-256(previous_hash || current_record), making the entire log
tamper-evident.  If any row is altered the chain breaks and
``verify_chain()`` reports the exact break point.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.constants import DEFAULT_RETENTION_DAYS

_GENESIS_HASH = "0" * 64  # chain seed


@dataclass
class AuditRecord:
    """Single compliance event."""

    event_type: str
    timestamp: float
    actor: str
    detail: dict[str, Any]
    input_hash: str = ""
    model_version: str = ""
    decision: str = ""
    oversight_action: str = "none"
    trace_id: str = ""
    chain_hash: str = ""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def _hash_record(previous_hash: str, rec: AuditRecord) -> str:
    """SHA-256(previous_hash || deterministic JSON of record)."""
    payload = {
        "record_id": rec.record_id,
        "event_type": rec.event_type,
        "timestamp": rec.timestamp,
        "actor": rec.actor,
        "detail": rec.detail,
        "input_hash": rec.input_hash,
        "model_version": rec.model_version,
        "decision": rec.decision,
        "oversight_action": rec.oversight_action,
        "trace_id": rec.trace_id,
    }
    blob = previous_hash + json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ComplianceAuditLog:
    """Append-only, hash-chained audit log.  Art. 12 compliant."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        db_path: Path,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.db_path = db_path
        self.retention_days = retention_days
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id  TEXT    NOT NULL,
                event_type TEXT    NOT NULL,
                timestamp  REAL   NOT NULL,
                actor      TEXT   NOT NULL,
                detail     TEXT   NOT NULL,
                input_hash TEXT,
                model_version TEXT,
                decision   TEXT,
                oversight_action TEXT DEFAULT 'none',
                trace_id   TEXT,
                chain_hash TEXT   NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts
                ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_evt
                ON audit_log(event_type);
            CREATE INDEX IF NOT EXISTS idx_audit_trace
                ON audit_log(trace_id);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        event_type: str,
        actor: str,
        detail: dict[str, Any],
        *,
        input_hash: str = "",
        model_version: str = "",
        decision: str = "",
        oversight_action: str = "none",
        trace_id: str = "",
    ) -> str:
        """Append an event.  Returns the record_id."""
        assert self._conn is not None
        rec = AuditRecord(
            event_type=event_type,
            timestamp=time.time(),
            actor=actor,
            detail=detail,
            input_hash=input_hash,
            model_version=model_version,
            decision=decision,
            oversight_action=oversight_action,
            trace_id=trace_id,
        )

        prev = self._last_chain_hash()
        rec.chain_hash = _hash_record(prev, rec)

        self._conn.execute(
            """INSERT INTO audit_log
               (record_id, event_type, timestamp, actor, detail,
                input_hash, model_version, decision,
                oversight_action, trace_id, chain_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.record_id,
                rec.event_type,
                rec.timestamp,
                rec.actor,
                json.dumps(rec.detail, default=str),
                rec.input_hash,
                rec.model_version,
                rec.decision,
                rec.oversight_action,
                rec.trace_id,
                rec.chain_hash,
            ),
        )
        self._conn.commit()
        return rec.record_id

    # ------------------------------------------------------------------
    # Read / verify
    # ------------------------------------------------------------------

    def verify_chain(self) -> dict[str, Any]:
        """Walk entire chain, verify no tampering."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT record_id, event_type, timestamp, actor, detail,"
            " input_hash, model_version, decision,"
            " oversight_action, trace_id, chain_hash"
            " FROM audit_log ORDER BY id"
        )
        prev = _GENESIS_HASH
        checked = 0
        for row in cur:
            rec = AuditRecord(
                event_type=row[1],
                timestamp=row[2],
                actor=row[3],
                detail=json.loads(row[4]),
                input_hash=row[5] or "",
                model_version=row[6] or "",
                decision=row[7] or "",
                oversight_action=row[8] or "none",
                trace_id=row[9] or "",
                record_id=row[0],
            )
            expected = _hash_record(prev, rec)
            stored = row[10]
            if expected != stored:
                return {
                    "valid": False,
                    "records_checked": checked,
                    "first_break": rec.record_id,
                }
            prev = stored
            checked += 1
        return {"valid": True, "records_checked": checked}

    def export(
        self, start: float, end: float,
    ) -> list[dict[str, Any]]:
        """Export records in a time range (for regulator requests)."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT record_id, event_type, timestamp, actor, detail,"
            " input_hash, model_version, decision,"
            " oversight_action, trace_id, chain_hash"
            " FROM audit_log WHERE timestamp BETWEEN ? AND ?"
            " ORDER BY id",
            (start, end),
        )
        rows: list[dict[str, Any]] = []
        for r in cur:
            rows.append(
                {
                    "record_id": r[0],
                    "event_type": r[1],
                    "timestamp": r[2],
                    "actor": r[3],
                    "detail": json.loads(r[4]),
                    "input_hash": r[5],
                    "model_version": r[6],
                    "decision": r[7],
                    "oversight_action": r[8],
                    "trace_id": r[9],
                    "chain_hash": r[10],
                }
            )
        return rows

    def retention_sweep(self) -> int:
        """Delete records past retention period.  Returns count."""
        assert self._conn is not None
        cutoff = time.time() - self.retention_days * 86400
        cur = self._conn.execute(
            "DELETE FROM audit_log WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        """Summary statistics for status reporting."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp)"
            " FROM audit_log"
        )
        total, first_ts, last_ts = cur.fetchone()
        return {
            "total_records": total,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "retention_days": self.retention_days,
            "db_path": str(self.db_path),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _last_chain_hash(self) -> str:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT chain_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else _GENESIS_HASH

    def as_dict(self, rec: AuditRecord) -> dict[str, Any]:
        """Serialize an AuditRecord to dict."""
        return asdict(rec)
