"""
Art. 9 — Risk management system.

Lifecycle risk registry backed by SQLite.  Risks are registered,
assessed, mitigated, and monitored.  Telemetry can be fed in to
auto-flag new risks (Art. 72 post-market monitoring hook).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog


@dataclass
class Risk:
    """A single risk entry."""

    id: str
    description: str
    severity: int          # 0-3 (RISK_MINIMAL..RISK_UNACCEPTABLE)
    likelihood: float      # 0.0-1.0
    mitigation: str
    status: str            # "open", "mitigated", "accepted", "closed"
    created: float
    updated: float
    residual_risk: str     # what remains after mitigation


class RiskRegistry:
    """Continuous lifecycle risk management.  Art. 9 compliant."""

    def __init__(
        self,
        audit_log: ComplianceAuditLog,
        db_path: Path,
    ) -> None:
        self._audit = audit_log
        self.db_path = db_path
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
            CREATE TABLE IF NOT EXISTS risks (
                id            TEXT PRIMARY KEY,
                description   TEXT NOT NULL,
                severity      INTEGER NOT NULL,
                likelihood    REAL NOT NULL,
                mitigation    TEXT NOT NULL DEFAULT '',
                status        TEXT NOT NULL DEFAULT 'open',
                created       REAL NOT NULL,
                updated       REAL NOT NULL,
                residual_risk TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        description: str,
        severity: int,
        likelihood: float = 0.5,
        mitigation: str = "",
        residual_risk: str = "",
    ) -> str:
        """Register a new risk.  Logged.  Returns risk_id."""
        assert self._conn is not None
        rid = uuid.uuid4().hex[:12]
        now = time.time()
        self._conn.execute(
            "INSERT INTO risks VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rid, description, severity, likelihood,
                mitigation, "open", now, now, residual_risk,
            ),
        )
        self._conn.commit()
        self._audit.record(
            event_type="risk_registered",
            actor="system",
            detail={
                "risk_id": rid,
                "description": description,
                "severity": severity,
                "likelihood": likelihood,
            },
        )
        return rid

    def update(
        self, risk_id: str, **changes: Any,
    ) -> dict[str, Any]:
        """Update risk assessment.  Logged."""
        assert self._conn is not None
        allowed = {
            "description", "severity", "likelihood",
            "mitigation", "status", "residual_risk",
        }
        updates = {
            k: v for k, v in changes.items() if k in allowed
        }
        if not updates:
            return {"error": "no valid fields to update"}

        updates["updated"] = time.time()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [risk_id]
        self._conn.execute(
            f"UPDATE risks SET {set_clause} WHERE id=?",  # noqa: S608
            vals,
        )
        self._conn.commit()
        self._audit.record(
            event_type="risk_updated",
            actor="system",
            detail={"risk_id": risk_id, "changes": updates},
        )
        return {"risk_id": risk_id, "updated_fields": list(updates)}

    def get(self, risk_id: str) -> Risk | None:
        """Fetch a single risk by ID."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT * FROM risks WHERE id=?", (risk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return Risk(*row)

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self) -> dict[str, Any]:
        """Current risk posture — open risks by severity."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT severity, COUNT(*) FROM risks"
            " WHERE status='open' GROUP BY severity"
        )
        by_severity = {row[0]: row[1] for row in cur.fetchall()}
        cur2 = self._conn.execute(
            "SELECT COUNT(*) FROM risks WHERE status='open'"
        )
        total_open = cur2.fetchone()[0]
        return {
            "open_risks": total_open,
            "by_severity": by_severity,
        }

    def residual_report(self) -> dict[str, Any]:
        """Art. 9 residual risk documentation."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, description, severity, residual_risk"
            " FROM risks WHERE status IN ('mitigated','accepted')"
        )
        items = [
            {
                "id": r[0],
                "description": r[1],
                "severity": r[2],
                "residual_risk": r[3],
            }
            for r in cur.fetchall()
        ]
        return {"residual_risks": items, "count": len(items)}

    # ------------------------------------------------------------------
    # Telemetry feed (Art. 72 hook)
    # ------------------------------------------------------------------

    def feed_telemetry(
        self, data: dict[str, Any],
    ) -> list[str]:
        """Ingest monitoring data, auto-flag new risks.

        Returns list of newly created risk IDs.
        """
        new_ids: list[str] = []

        # Example heuristic: error_rate above threshold
        error_rate = data.get("error_rate", 0.0)
        if isinstance(error_rate, int | float) and error_rate > 0.1:
            rid = self.register(
                description=(
                    f"Elevated error rate detected: {error_rate:.2%}"
                ),
                severity=2,
                likelihood=0.8,
                mitigation="Investigate root cause",
            )
            new_ids.append(rid)

        # Latency degradation
        latency_ms = data.get("avg_latency_ms", 0)
        if isinstance(latency_ms, int | float) and latency_ms > 5000:
            rid = self.register(
                description=(
                    f"High latency detected: {latency_ms:.0f}ms"
                ),
                severity=1,
                likelihood=0.6,
                mitigation="Check resource utilisation",
            )
            new_ids.append(rid)

        if new_ids:
            self._audit.record(
                event_type="telemetry_risks_flagged",
                actor="system",
                detail={
                    "telemetry": json.dumps(data, default=str),
                    "new_risk_ids": new_ids,
                },
            )
        return new_ids
