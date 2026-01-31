"""
Art. 17 — Quality management system.

Code-side support for QMS processes:
- Change management (record, approve, list)
- Version tracking per component
- Internal audit scheduling and completion
- Non-conformity tracking (linked to incidents)
- Training records

All mutations are logged through ComplianceAuditLog.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog


class QualityManagementSystem:
    """Art. 17 — QMS process tracking."""

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
            CREATE TABLE IF NOT EXISTS changes (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                author TEXT NOT NULL,
                risk_assessment TEXT NOT NULL,
                files_changed TEXT NOT NULL,
                approved_by TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                created REAL NOT NULL,
                updated REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS versions (
                component TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                changelog TEXT NOT NULL,
                updated REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audits (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                auditor TEXT NOT NULL,
                due_date REAL NOT NULL,
                status TEXT DEFAULT 'scheduled',
                findings TEXT DEFAULT '[]',
                completed_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS nonconformities (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                severity INTEGER NOT NULL,
                incident_id TEXT DEFAULT '',
                corrective_action TEXT DEFAULT '',
                resolution TEXT DEFAULT '',
                resolver TEXT DEFAULT '',
                status TEXT DEFAULT 'open',
                created REAL NOT NULL,
                resolved_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS training (
                id TEXT PRIMARY KEY,
                trainee TEXT NOT NULL,
                subject TEXT NOT NULL,
                date REAL NOT NULL,
                trainer TEXT DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Change management
    # ------------------------------------------------------------------

    def record_change(
        self,
        description: str,
        author: str,
        risk_assessment: str,
        files_changed: list[str],
        approved_by: str = "",
    ) -> str:
        """Record a system change.  Returns change_id."""
        assert self._conn is not None
        cid = f"CHG-{uuid.uuid4().hex[:8]}"
        now = time.time()
        status = "approved" if approved_by else "pending"
        self._conn.execute(
            "INSERT INTO changes VALUES (?,?,?,?,?,?,?,?,?)",
            (
                cid, description, author, risk_assessment,
                json.dumps(files_changed),
                approved_by, status, now, now,
            ),
        )
        self._conn.commit()
        self._audit.record(
            event_type="system_change",
            actor=author,
            detail={
                "change_id": cid,
                "description": description,
                "risk_assessment": risk_assessment,
                "files_changed": files_changed,
                "status": status,
            },
        )
        return cid

    def approve_change(
        self, change_id: str, approver: str,
    ) -> dict[str, Any]:
        """Approve a pending change."""
        assert self._conn is not None
        now = time.time()
        self._conn.execute(
            "UPDATE changes SET approved_by=?, status='approved',"
            " updated=? WHERE id=?",
            (approver, now, change_id),
        )
        self._conn.commit()
        self._audit.record(
            event_type="change_approved",
            actor=approver,
            detail={"change_id": change_id},
            oversight_action="approve",
        )
        return {
            "change_id": change_id,
            "approved_by": approver,
            "status": "approved",
        }

    def list_changes(
        self, status: str = "all",
    ) -> list[dict[str, Any]]:
        """List changes, optionally filtered by status."""
        assert self._conn is not None
        if status == "all":
            cur = self._conn.execute(
                "SELECT * FROM changes ORDER BY created DESC"
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM changes WHERE status=?"
                " ORDER BY created DESC",
                (status,),
            )
        return [
            {
                "id": r[0],
                "description": r[1],
                "author": r[2],
                "risk_assessment": r[3],
                "files_changed": json.loads(r[4]),
                "approved_by": r[5],
                "status": r[6],
                "created": r[7],
                "updated": r[8],
            }
            for r in cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Version tracking
    # ------------------------------------------------------------------

    def record_version(
        self,
        component: str,
        version: str,
        changelog: str,
    ) -> None:
        """Record current version of a component."""
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO versions VALUES (?,?,?,?)",
            (component, version, changelog, time.time()),
        )
        self._conn.commit()
        self._audit.record(
            event_type="version_recorded",
            actor="system",
            detail={
                "component": component,
                "version": version,
                "changelog": changelog,
            },
        )

    def current_versions(self) -> dict[str, str]:
        """Current version of each tracked component."""
        assert self._conn is not None
        cur = self._conn.execute("SELECT component, version FROM versions")
        return {r[0]: r[1] for r in cur.fetchall()}

    # ------------------------------------------------------------------
    # Internal audits
    # ------------------------------------------------------------------

    def schedule_audit(
        self,
        scope: str,
        auditor: str,
        due_date: float,
    ) -> str:
        """Schedule an internal audit.  Returns audit_id."""
        assert self._conn is not None
        aid = f"AUD-{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO audits (id, scope, auditor, due_date)"
            " VALUES (?,?,?,?)",
            (aid, scope, auditor, due_date),
        )
        self._conn.commit()
        self._audit.record(
            event_type="audit_scheduled",
            actor=auditor,
            detail={
                "audit_id": aid,
                "scope": scope,
                "due_date": due_date,
            },
        )
        return aid

    def complete_audit(
        self,
        audit_id: str,
        findings: list[dict[str, Any]],
        auditor: str,
    ) -> dict[str, Any]:
        """Complete an audit with findings."""
        assert self._conn is not None
        now = time.time()
        self._conn.execute(
            "UPDATE audits SET status='completed',"
            " findings=?, completed_at=? WHERE id=?",
            (json.dumps(findings, default=str), now, audit_id),
        )
        self._conn.commit()
        result: dict[str, Any] = {
            "audit_id": audit_id,
            "status": "completed",
            "findings_count": len(findings),
            "completed_at": now,
        }
        self._audit.record(
            event_type="audit_completed",
            actor=auditor,
            detail={**result, "findings": findings},
        )
        return result

    def list_audits(
        self, status: str = "all",
    ) -> list[dict[str, Any]]:
        """List audits, optionally filtered by status."""
        assert self._conn is not None
        if status == "all":
            cur = self._conn.execute(
                "SELECT * FROM audits ORDER BY due_date DESC"
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM audits WHERE status=?"
                " ORDER BY due_date DESC",
                (status,),
            )
        return [
            {
                "id": r[0],
                "scope": r[1],
                "auditor": r[2],
                "due_date": r[3],
                "status": r[4],
                "findings": json.loads(r[5]),
                "completed_at": r[6],
            }
            for r in cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Non-conformity tracking
    # ------------------------------------------------------------------

    def record_nonconformity(
        self,
        description: str,
        severity: int,
        incident_id: str = "",
        corrective_action: str = "",
    ) -> str:
        """Record a non-conformity.  Returns nc_id."""
        assert self._conn is not None
        nc_id = f"NC-{uuid.uuid4().hex[:8]}"
        now = time.time()
        self._conn.execute(
            "INSERT INTO nonconformities"
            " (id, description, severity, incident_id,"
            "  corrective_action, status, created)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                nc_id, description, severity,
                incident_id, corrective_action, "open", now,
            ),
        )
        self._conn.commit()
        self._audit.record(
            event_type="nonconformity_recorded",
            actor="system",
            detail={
                "nc_id": nc_id,
                "description": description,
                "severity": severity,
                "incident_id": incident_id,
            },
        )
        return nc_id

    def resolve_nonconformity(
        self,
        nc_id: str,
        resolution: str,
        resolver: str,
    ) -> dict[str, Any]:
        """Resolve a non-conformity."""
        assert self._conn is not None
        now = time.time()
        self._conn.execute(
            "UPDATE nonconformities SET resolution=?, resolver=?,"
            " status='resolved', resolved_at=? WHERE id=?",
            (resolution, resolver, now, nc_id),
        )
        self._conn.commit()
        result: dict[str, Any] = {
            "nc_id": nc_id,
            "resolution": resolution,
            "resolver": resolver,
            "status": "resolved",
            "resolved_at": now,
        }
        self._audit.record(
            event_type="nonconformity_resolved",
            actor=resolver,
            detail=result,
        )
        return result

    # ------------------------------------------------------------------
    # Training records
    # ------------------------------------------------------------------

    def record_training(
        self,
        trainee: str,
        subject: str,
        date: float,
        trainer: str = "",
    ) -> str:
        """Record a training completion.  Returns training_id."""
        assert self._conn is not None
        tid = f"TRN-{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            "INSERT INTO training VALUES (?,?,?,?,?)",
            (tid, trainee, subject, date, trainer),
        )
        self._conn.commit()
        self._audit.record(
            event_type="training_recorded",
            actor=trainer or "system",
            detail={
                "training_id": tid,
                "trainee": trainee,
                "subject": subject,
                "date": date,
            },
        )
        return tid

    def training_status(self) -> dict[str, Any]:
        """Summary of training records."""
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT trainee),"
            " COUNT(DISTINCT subject) FROM training"
        )
        total, trainees, subjects = cur.fetchone()
        return {
            "total_records": total,
            "unique_trainees": trainees,
            "unique_subjects": subjects,
        }
