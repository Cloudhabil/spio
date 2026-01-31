"""
Art. 11 — Technical documentation (Annex IV).

Generates the formal Annex IV technical documentation required by
the EU AI Act.  Pulls data from runtime status, risk registry,
audit log, hardware constants, and transparency report.

Each ``section_*`` method maps to a numbered Annex IV section.
``generate()`` assembles the full document as a structured dict
that can be exported to JSON for submission.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog
from sovereign_pio.compliance.risk_management import RiskRegistry
from sovereign_pio.compliance.transparency import TransparencyReport
from sovereign_pio.constants import (
    DIMENSION_NAMES,
    DIMENSION_SILICON,
    LUCAS_NUMBERS,
    TOTAL_STATES,
)


class AnnexIVGenerator:
    """Generate EU AI Act Annex IV technical documentation."""

    VERSION = "2.0.0"  # Phase 2

    def __init__(
        self,
        audit_log: ComplianceAuditLog,
        risk_registry: RiskRegistry,
        transparency: TransparencyReport,
    ) -> None:
        self._audit = audit_log
        self._risk = risk_registry
        self._transparency = transparency

    # ------------------------------------------------------------------
    # Section 1 — General description
    # ------------------------------------------------------------------
    def section_1_general_description(self) -> dict[str, Any]:
        """Intended purpose, developer identity, version, date."""
        return {
            "system_name": "Sovereign PIO (SPIO)",
            "developer": "Brahim / Sovereign PIO Project",
            "version": self.VERSION,
            "date": time.strftime("%Y-%m-%d"),
            "intended_purpose": (
                "Personal Intelligent Operator — AI assistant with "
                "deterministic safety constraints, 12-dimension "
                "architecture, and EU AI Act compliance."
            ),
            "risk_classification": "high-risk",
            "regulation": "EU AI Act (Regulation 2024/1689)",
        }

    # ------------------------------------------------------------------
    # Section 2 — System elements and development process
    # ------------------------------------------------------------------
    def section_2_system_elements(self) -> dict[str, Any]:
        """Architecture, algorithms, data processing, hardware."""
        return {
            "architecture": {
                "layers": [
                    "PIO (interface)",
                    "GPIA (reasoning)",
                    "ASIOS (governance)",
                    "Moltbot (multi-channel gateway)",
                ],
                "dimensions": 12,
                "total_states": TOTAL_STATES,
                "lucas_capacity": LUCAS_NUMBERS,
            },
            "algorithms": [
                "PHI-deterministic dimension routing",
                "GuardianGate safety classification",
                "Hash-chained audit logging",
                "BIL provenance tracking",
            ],
            "data_processing": (
                "All inputs pass through GuardianGate (Art. 14) "
                "before reaching intelligence layer. "
                "Data tagged with BIL provenance labels (Art. 10)."
            ),
            "development_methodology": "Iterative with compliance gates",
        }

    # ------------------------------------------------------------------
    # Section 3 — Monitoring, functioning, control
    # ------------------------------------------------------------------
    def section_3_monitoring_control(self) -> dict[str, Any]:
        """Human oversight capabilities, logging, alerting."""
        return {
            "monitoring": {
                "audit_log": "Hash-chained SQLite (Art. 12)",
                "post_market": "Continuous metric ingestion (Art. 72)",
                "slo_tracking": "Service Level Objectives with auto-escalation",
            },
            "human_oversight": {
                "guardian_gate": "Tier-1 safety gate on all inputs",
                "oversight_interface": "Human override / confirmation flow",
                "stop_switch": "Emergency halt with state preservation",
            },
            "alerting": {
                "drift_detection": "Statistical comparison to baseline",
                "slo_breach": "Auto-escalation to risk registry / incidents",
                "chain_integrity": "Tamper-evident hash chain verification",
            },
        }

    # ------------------------------------------------------------------
    # Section 4 — Risk management
    # ------------------------------------------------------------------
    def section_4_risk_management(self) -> dict[str, Any]:
        """From RiskRegistry — full risk posture + residual."""
        posture = self._risk.assess()
        residual = self._risk.residual_report()
        return {
            "risk_posture": posture,
            "residual_risks": residual,
            "methodology": (
                "Continuous lifecycle risk management per Art. 9. "
                "Risks are registered, assessed, mitigated, and monitored. "
                "Telemetry feeds auto-flag new risks."
            ),
        }

    # ------------------------------------------------------------------
    # Section 5 — Lifecycle changes
    # ------------------------------------------------------------------
    def section_5_lifecycle_changes(self) -> dict[str, Any]:
        """Version history from audit log change events."""
        recent = self._audit.export(
            start=0.0,
            end=time.time(),
        )
        changes = [
            r for r in recent
            if r["event_type"] in (
                "risk_registered", "risk_updated",
                "source_upgraded", "system_change",
            )
        ]
        return {
            "total_audit_records": len(recent),
            "change_events": len(changes),
            "latest_changes": changes[-10:] if changes else [],
            "note": (
                "Full audit trail available via "
                "ComplianceAuditLog.export()."
            ),
        }

    # ------------------------------------------------------------------
    # Section 6 — Standards applied
    # ------------------------------------------------------------------
    def section_6_standards_applied(self) -> dict[str, Any]:
        """Harmonised standards referenced."""
        return {
            "standards": [
                {
                    "id": "ISO/IEC 42001:2023",
                    "title": "AI Management System",
                    "relevance": "QMS framework",
                },
                {
                    "id": "IEC 62443",
                    "title": "Industrial Cybersecurity",
                    "relevance": "Robustness (Art. 15)",
                },
                {
                    "id": "ISO/IEC 27001:2022",
                    "title": "Information Security Management",
                    "relevance": "Data governance (Art. 10)",
                },
                {
                    "id": "ISO 14971:2019",
                    "title": "Risk Management for Medical Devices",
                    "relevance": "Risk methodology (Art. 9)",
                },
                {
                    "id": "ISO/IEC 25010:2023",
                    "title": "Software Quality Model",
                    "relevance": "Accuracy, robustness (Art. 15)",
                },
            ],
        }

    # ------------------------------------------------------------------
    # Section 7 — Data requirements
    # ------------------------------------------------------------------
    def section_7_data_requirements(self) -> dict[str, Any]:
        """BIL source distribution, sensitivity classification stats."""
        return {
            "data_provenance": "BIL (Brahim Industry Label) system",
            "bil_format": "BIL:<sector>:<type>:<source>:<id>-<check>",
            "source_types": {
                "deterministic": "source < 900 (human-verified)",
                "ml_predicted": "source = 900",
                "unverified": "source = 999",
            },
            "sensitivity_levels": {
                0: "Public",
                1: "Internal",
                2: "Confidential",
                3: "Restricted (PII, health, financial)",
            },
            "governance_note": (
                "All data elements tagged with BIL labels before "
                "processing. ML predictions flagged for human review."
            ),
        }

    # ------------------------------------------------------------------
    # Section 8 — Human oversight measures
    # ------------------------------------------------------------------
    def section_8_human_oversight(self) -> dict[str, Any]:
        """GuardianGate config, override history, stop switch status."""
        audit_stats = self._audit.stats()
        return {
            "guardian_gate": {
                "description": "Deterministic safety gate on every input",
                "timeout_ms": 500,
                "actions": ["allow", "block", "escalate"],
            },
            "oversight_interface": {
                "description": "Human override and confirmation flow",
                "capabilities": [
                    "Override any system decision",
                    "Require confirmation for high-risk actions",
                    "List pending confirmations",
                ],
            },
            "stop_switch": {
                "description": "Emergency halt with state preservation",
                "capabilities": [
                    "Immediate halt of all processing",
                    "State snapshot for incident investigation",
                    "Resume only by authorised human",
                ],
            },
            "audit_coverage": audit_stats,
        }

    # ------------------------------------------------------------------
    # Section 9 — Computational resources
    # ------------------------------------------------------------------
    def section_9_computational_resources(self) -> dict[str, Any]:
        """Hardware: NPU/GPU/CPU from DIMENSION_SILICON + measured BW."""
        dimension_hw: list[dict[str, Any]] = []
        for dim in range(1, 13):
            dimension_hw.append({
                "dimension": dim,
                "name": DIMENSION_NAMES.get(dim, ""),
                "silicon": DIMENSION_SILICON.get(dim, "CPU"),
                "lucas_states": LUCAS_NUMBERS[dim - 1],
            })
        return {
            "silicon_targets": ["NPU", "CPU", "GPU"],
            "measured_bandwidths": {
                "NPU": "7.35 GB/s (k=PHI)",
                "GPU": "12.0 GB/s (BW=PHI*NPU)",
                "RAM": "26.0 GB/s (baseline)",
                "SSD": "2.8 GB/s (BW=NPU/PHI^2)",
            },
            "dimension_mapping": dimension_hw,
            "total_states": TOTAL_STATES,
        }

    # ------------------------------------------------------------------
    # Section 10 — Known limitations
    # ------------------------------------------------------------------
    def section_10_limitations(self) -> dict[str, Any]:
        """Known limitations + foreseeable risks from risk registry."""
        limitations = self._transparency.known_limitations()
        posture = self._risk.assess()
        return {
            "known_limitations": limitations.get("general", []),
            "risk_posture": posture,
            "foreseeable_misuse": [
                "Using system outputs for autonomous legal judgements",
                "Relying on system for real-time medical diagnosis",
                "Operating without human oversight in critical decisions",
            ],
            "mitigation": (
                "GuardianGate blocks or escalates requests that "
                "fall outside intended purpose."
            ),
        }

    # ------------------------------------------------------------------
    # Full document
    # ------------------------------------------------------------------
    def generate(self) -> dict[str, Any]:
        """Full Annex IV document as structured dict."""
        doc: dict[str, Any] = {
            "annex_iv_version": self.VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regulation": "EU AI Act (Regulation 2024/1689) — Annex IV",
            "section_1_general_description": (
                self.section_1_general_description()
            ),
            "section_2_system_elements": (
                self.section_2_system_elements()
            ),
            "section_3_monitoring_control": (
                self.section_3_monitoring_control()
            ),
            "section_4_risk_management": (
                self.section_4_risk_management()
            ),
            "section_5_lifecycle_changes": (
                self.section_5_lifecycle_changes()
            ),
            "section_6_standards_applied": (
                self.section_6_standards_applied()
            ),
            "section_7_data_requirements": (
                self.section_7_data_requirements()
            ),
            "section_8_human_oversight": (
                self.section_8_human_oversight()
            ),
            "section_9_computational_resources": (
                self.section_9_computational_resources()
            ),
            "section_10_limitations": (
                self.section_10_limitations()
            ),
        }
        self._audit.record(
            event_type="annex_iv_generated",
            actor="system",
            detail={
                "sections": [k for k in doc if k.startswith("section_")],
                "version": self.VERSION,
            },
        )
        return doc

    def export_json(self, path: Path) -> None:
        """Write Annex IV to JSON file for submission."""
        doc = self.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, default=str))
