"""
Art. 73 — Reporting of serious incidents.

IncidentFreezer snapshots system state when a serious incident is
detected.  IncidentReport tracks structured reports with deadline
enforcement per Art. 73.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.constants import (
    INCIDENT_DEADLINE_CRITICAL_INFRA_H,
    INCIDENT_DEADLINE_DEATH_H,
    INCIDENT_DEADLINE_OTHER_H,
)
from sovereign_pio.compliance.record_keeping import ComplianceAuditLog

_CATEGORY_DEADLINE_H: dict[str, int] = {
    "critical_infra": INCIDENT_DEADLINE_CRITICAL_INFRA_H,
    "death_or_health": INCIDENT_DEADLINE_DEATH_H,
    "other": INCIDENT_DEADLINE_OTHER_H,
}


class IncidentFreezer:
    """Freeze system state on serious incident detection."""

    def __init__(
        self,
        audit_log: ComplianceAuditLog,
        archive_dir: Path,
    ) -> None:
        self._audit = audit_log
        self._archive_dir = archive_dir
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    def freeze(
        self,
        incident_type: str,
        description: str,
        actor: str,
    ) -> str:
        """Snapshot logs, config, model version, active state.

        Returns incident_id.  System state must not be altered
        until authorities are notified.
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8]}"
        ts = time.time()

        # Create incident directory
        inc_dir = self._archive_dir / incident_id
        inc_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot audit log
        audit_stats = self._audit.stats()
        db_path = Path(audit_stats["db_path"])
        if db_path.exists():
            shutil.copy2(str(db_path), str(inc_dir / "audit_log.db"))

        # Write incident manifest
        manifest = {
            "incident_id": incident_id,
            "incident_type": incident_type,
            "description": description,
            "actor": actor,
            "timestamp": ts,
            "audit_stats": audit_stats,
            "chain_integrity": self._audit.verify_chain(),
        }
        manifest_path = inc_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
        )

        self._audit.record(
            event_type="incident_frozen",
            actor=actor,
            detail={
                "incident_id": incident_id,
                "incident_type": incident_type,
                "description": description,
                "archive_path": str(inc_dir),
            },
            decision="freeze",
        )
        return incident_id

    def export(self, incident_id: str) -> Path:
        """Export sealed archive for regulator."""
        inc_dir = self._archive_dir / incident_id
        if not inc_dir.exists():
            raise FileNotFoundError(
                f"No archive for incident {incident_id}"
            )
        return inc_dir


class IncidentReport:
    """Structured incident reporting per Art. 73 deadlines."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._reports: dict[str, dict[str, Any]] = {}

    def create(
        self,
        incident_id: str,
        category: str,
        description: str,
    ) -> dict[str, Any]:
        """Create report.  Sets deadline based on category."""
        deadline_h = _CATEGORY_DEADLINE_H.get(
            category, INCIDENT_DEADLINE_OTHER_H,
        )
        now = time.time()
        report: dict[str, Any] = {
            "incident_id": incident_id,
            "category": category,
            "description": description,
            "created": now,
            "deadline": now + deadline_h * 3600,
            "deadline_hours": deadline_h,
            "status": "open",
            "supplements": [],
        }
        self._reports[incident_id] = report
        self._audit.record(
            event_type="incident_report_created",
            actor="system",
            detail=report,
        )
        return report

    def submit(self, incident_id: str) -> dict[str, Any]:
        """Mark as submitted.  Logged."""
        report = self._reports.get(incident_id)
        if report is None:
            return {"error": "unknown incident_id"}
        report["status"] = "submitted"
        report["submitted_at"] = time.time()
        self._audit.record(
            event_type="incident_report_submitted",
            actor="system",
            detail={"incident_id": incident_id},
        )
        return report

    def supplement(
        self, incident_id: str, additional: dict[str, Any],
    ) -> dict[str, Any]:
        """Add supplemental information.  Allowed by Art. 73."""
        report = self._reports.get(incident_id)
        if report is None:
            return {"error": "unknown incident_id"}
        entry = {
            "timestamp": time.time(),
            "data": additional,
        }
        report["supplements"].append(entry)
        self._audit.record(
            event_type="incident_report_supplemented",
            actor="system",
            detail={
                "incident_id": incident_id,
                "supplement": entry,
            },
        )
        return report

    def list_open(self) -> list[dict[str, Any]]:
        """Open incidents with deadline status."""
        now = time.time()
        result: list[dict[str, Any]] = []
        for report in self._reports.values():
            if report["status"] == "open":
                remaining_h = (
                    (report["deadline"] - now) / 3600
                )
                result.append({
                    "incident_id": report["incident_id"],
                    "category": report["category"],
                    "deadline_hours_remaining": round(
                        remaining_h, 1,
                    ),
                    "overdue": remaining_h < 0,
                })
        return result
