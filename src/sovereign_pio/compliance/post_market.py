"""
Art. 72 — Post-market monitoring.

Extends RiskRegistry telemetry into a proper monitoring system with:
- SQLite time-series metric storage
- SLO (Service Level Objective) tracking
- Statistical drift detection against baselines
- Auto-escalation: SLO breaches → new risks / incidents
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog
from sovereign_pio.compliance.risk_management import RiskRegistry


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str             # "guardian_latency_p99"
    metric: str           # "guardian_latency_ms"
    threshold: float      # 500.0
    comparator: str       # "lt" | "gt" | "eq"
    window_seconds: int   # 3600


class PostMarketMonitor:
    """Art. 72 — continuous post-market monitoring."""

    def __init__(
        self,
        audit_log: ComplianceAuditLog,
        risk_registry: RiskRegistry,
        db_path: Path,
    ) -> None:
        self._audit = audit_log
        self._risk = risk_registry
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
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                tags TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_name_ts
                ON metrics(metric, timestamp);

            CREATE TABLE IF NOT EXISTS baselines (
                metric TEXT PRIMARY KEY,
                mean REAL NOT NULL,
                std REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                updated REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS slos (
                name TEXT PRIMARY KEY,
                metric TEXT NOT NULL,
                threshold REAL NOT NULL,
                comparator TEXT NOT NULL,
                window_seconds INTEGER NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Metrics ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        metric: str,
        value: float,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """Store a single metric data point."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT INTO metrics (metric, value, timestamp, tags)"
            " VALUES (?, ?, ?, ?)",
            (
                metric,
                value,
                time.time(),
                json.dumps(tags or {}, default=str),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # SLO management
    # ------------------------------------------------------------------

    def register_slo(self, slo: SLO) -> None:
        """Register or update a Service Level Objective."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO slos VALUES (?, ?, ?, ?, ?)",
            (
                slo.name,
                slo.metric,
                slo.threshold,
                slo.comparator,
                slo.window_seconds,
            ),
        )
        self._conn.commit()
        self._audit.record(
            event_type="slo_registered",
            actor="system",
            detail={
                "name": slo.name,
                "metric": slo.metric,
                "threshold": slo.threshold,
                "comparator": slo.comparator,
                "window_seconds": slo.window_seconds,
            },
        )

    def check_slos(self) -> list[dict[str, Any]]:
        """Evaluate all SLOs against recent data.  Returns breaches."""
        assert self._conn is not None
        now = time.time()
        cur = self._conn.execute("SELECT * FROM slos")
        breaches: list[dict[str, Any]] = []
        for row in cur.fetchall():
            name, metric, threshold, comparator, window = row
            cutoff = now - window
            agg = self._conn.execute(
                "SELECT AVG(value), MAX(value), MIN(value), COUNT(*)"
                " FROM metrics WHERE metric=? AND timestamp>=?",
                (metric, cutoff),
            ).fetchone()
            avg, mx, mn, cnt = agg
            if cnt == 0:
                continue
            breached = self._check_comparator(
                avg, threshold, comparator,
            )
            if breached:
                breaches.append({
                    "slo_name": name,
                    "metric": metric,
                    "threshold": threshold,
                    "comparator": comparator,
                    "actual_avg": round(avg, 4),
                    "actual_max": round(mx, 4),
                    "actual_min": round(mn, 4),
                    "sample_count": cnt,
                    "window_seconds": window,
                    "breached_at": now,
                })
        return breaches

    @staticmethod
    def _check_comparator(
        value: float, threshold: float, comparator: str,
    ) -> bool:
        """Return True if the SLO is *breached*."""
        if comparator == "lt":
            return value >= threshold  # should be less than
        if comparator == "gt":
            return value <= threshold  # should be greater than
        if comparator == "eq":
            return abs(value - threshold) > 0.001
        return False

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def set_baseline(
        self, metric: str, mean: float, std: float,
    ) -> None:
        """Set or update the baseline for a metric."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO baselines VALUES (?, ?, ?, ?, ?)",
            (metric, mean, std, 0, time.time()),
        )
        self._conn.commit()

    def detect_drift(
        self,
        metric: str,
        window_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Compare recent window to baseline.

        Returns ``{drifted, shift, p_value}``.
        Uses z-score approximation for simplicity.
        """
        assert self._conn is not None
        # Get baseline
        cur = self._conn.execute(
            "SELECT mean, std FROM baselines WHERE metric=?",
            (metric,),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "drifted": False,
                "error": "no baseline set",
                "metric": metric,
            }
        baseline_mean, baseline_std = row

        # Get recent data
        cutoff = time.time() - window_seconds
        cur2 = self._conn.execute(
            "SELECT AVG(value), COUNT(*) FROM metrics"
            " WHERE metric=? AND timestamp>=?",
            (metric, cutoff),
        )
        recent_mean, count = cur2.fetchone()
        if count == 0 or recent_mean is None:
            return {
                "drifted": False,
                "error": "no recent data",
                "metric": metric,
            }

        # Z-score drift detection
        if baseline_std == 0:
            shift = abs(recent_mean - baseline_mean)
            drifted = shift > 0.001
            z_score = float("inf") if drifted else 0.0
        else:
            se = baseline_std / math.sqrt(max(count, 1))
            z_score = (recent_mean - baseline_mean) / se
            shift = abs(recent_mean - baseline_mean)
            drifted = abs(z_score) > 2.0  # 95% confidence

        # Approximate two-tailed p-value using error function
        p_value = math.erfc(abs(z_score) / math.sqrt(2))

        return {
            "drifted": drifted,
            "metric": metric,
            "baseline_mean": round(baseline_mean, 4),
            "recent_mean": round(recent_mean, 4),
            "shift": round(shift, 4),
            "z_score": round(z_score, 4),
            "p_value": round(p_value, 6),
            "sample_count": count,
            "window_seconds": window_seconds,
        }

    # ------------------------------------------------------------------
    # Auto-escalation
    # ------------------------------------------------------------------

    def evaluate(self) -> list[str]:
        """Run all checks.  Auto-creates risks/incidents for breaches.

        Returns list of new risk/incident IDs.
        """
        new_ids: list[str] = []

        # Check SLOs
        for breach in self.check_slos():
            rid = self._risk.register(
                description=(
                    f"SLO breach: {breach['slo_name']} — "
                    f"avg={breach['actual_avg']} vs "
                    f"threshold={breach['threshold']}"
                ),
                severity=2,
                likelihood=0.9,
                mitigation="Investigate SLO breach root cause",
            )
            new_ids.append(rid)

        # Check drift on all baselined metrics
        assert self._conn is not None
        cur = self._conn.execute("SELECT metric FROM baselines")
        for (metric,) in cur.fetchall():
            drift = self.detect_drift(metric)
            if drift.get("drifted"):
                rid = self._risk.register(
                    description=(
                        f"Drift detected on '{metric}': "
                        f"shift={drift.get('shift', 0)}, "
                        f"p={drift.get('p_value', 0)}"
                    ),
                    severity=1,
                    likelihood=0.7,
                    mitigation="Review metric trend and update baseline",
                )
                new_ids.append(rid)

        if new_ids:
            self._audit.record(
                event_type="post_market_evaluation",
                actor="system",
                detail={
                    "new_risk_ids": new_ids,
                    "slo_breaches": len(self.check_slos()),
                },
            )
        return new_ids

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        """Summary: SLO status, drift alerts, metric trends."""
        assert self._conn is not None
        breaches = self.check_slos()

        # Drift status for all baselined metrics
        drift_alerts: list[dict[str, Any]] = []
        cur = self._conn.execute("SELECT metric FROM baselines")
        for (metric,) in cur.fetchall():
            drift = self.detect_drift(metric)
            if drift.get("drifted"):
                drift_alerts.append(drift)

        # Recent metric counts
        now = time.time()
        hour_ago = now - 3600
        cur2 = self._conn.execute(
            "SELECT metric, COUNT(*), AVG(value)"
            " FROM metrics WHERE timestamp>=?"
            " GROUP BY metric",
            (hour_ago,),
        )
        recent_metrics: list[dict[str, Any]] = [
            {"metric": r[0], "count_1h": r[1], "avg_1h": round(r[2], 4)}
            for r in cur2.fetchall()
        ]

        return {
            "slo_breaches": breaches,
            "slo_breach_count": len(breaches),
            "drift_alerts": drift_alerts,
            "drift_alert_count": len(drift_alerts),
            "recent_metrics": recent_metrics,
            "checked_at": now,
        }
