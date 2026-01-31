"""
Art. 43+ — Conformity assessment preparation.

Provides:
- CE marking checklist (auto-detects which requirements are met)
- Declaration of conformity (Art. 47)
- EU database registration data (Art. 71)
- Assessment route determination (Annex VI vs VII)
- Gap analysis for remaining action items
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog


@dataclass
class ConformityRequirement:
    """A single conformity checklist item."""

    article: str          # "Art. 9"
    description: str
    status: str           # "met", "partial", "not_met", "not_applicable"
    evidence: str         # pointer to module/function that satisfies it
    last_checked: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Module-detection map: article → (module_path, evidence_string)
# ---------------------------------------------------------------------------
_REQUIREMENT_MODULES: list[tuple[str, str, str, str]] = [
    (
        "Art. 9",
        "Risk management system",
        "sovereign_pio.compliance.risk_management",
        "RiskRegistry — lifecycle risk management",
    ),
    (
        "Art. 10",
        "Data and data governance",
        "sovereign_pio.compliance.data_governance",
        "DataGovernance + BIL provenance tracking",
    ),
    (
        "Art. 11",
        "Technical documentation (Annex IV)",
        "sovereign_pio.compliance.annex_iv",
        "AnnexIVGenerator — 10-section Annex IV",
    ),
    (
        "Art. 12",
        "Record-keeping (audit logging)",
        "sovereign_pio.compliance.record_keeping",
        "ComplianceAuditLog — hash-chained SQLite",
    ),
    (
        "Art. 13",
        "Transparency and information to deployers",
        "sovereign_pio.compliance.transparency",
        "TransparencyReport — full Art. 13 disclosure",
    ),
    (
        "Art. 14",
        "Human oversight",
        "sovereign_pio.compliance.human_oversight",
        "GuardianGate + OversightInterface + StopSwitch",
    ),
    (
        "Art. 15",
        "Accuracy, robustness, and cybersecurity",
        "sovereign_pio.compliance.robustness",
        "InputValidator + ModelIntegrityVerifier",
    ),
    (
        "Art. 17",
        "Quality management system",
        "sovereign_pio.compliance.qms",
        "QualityManagementSystem — change/audit/NC/training",
    ),
    (
        "Art. 26",
        "Obligations of deployers (API)",
        "sovereign_pio.compliance",
        "substrate_gateway /compliance/* endpoints",
    ),
    (
        "Art. 72",
        "Post-market monitoring",
        "sovereign_pio.compliance.post_market",
        "PostMarketMonitor — SLO + drift detection",
    ),
    (
        "Art. 73",
        "Reporting of serious incidents",
        "sovereign_pio.compliance.incident",
        "IncidentFreezer + IncidentReport",
    ),
]


class ConformityAssessment:
    """Art. 43 — conformity assessment preparation."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------

    def checklist(self) -> list[ConformityRequirement]:
        """Full checklist of EU AI Act requirements with status.

        Auto-detects which requirements are met based on whether
        the corresponding Python module is importable.
        """
        now = time.time()
        results: list[ConformityRequirement] = []
        for article, desc, module_path, evidence in _REQUIREMENT_MODULES:
            try:
                importlib.import_module(module_path)
                status = "met"
            except ImportError:
                status = "not_met"
            results.append(
                ConformityRequirement(
                    article=article,
                    description=desc,
                    status=status,
                    evidence=evidence if status == "met" else "",
                    last_checked=now,
                )
            )
        self._audit.record(
            event_type="conformity_checklist",
            actor="system",
            detail={
                "total": len(results),
                "met": sum(1 for r in results if r.status == "met"),
                "not_met": sum(
                    1 for r in results if r.status == "not_met"
                ),
            },
        )
        return results

    # ------------------------------------------------------------------
    # Declaration of conformity (Art. 47)
    # ------------------------------------------------------------------

    def declaration_of_conformity(self) -> dict[str, Any]:
        """Generate Art. 47 declaration.

        Structured dict matching EU template: manufacturer, system ID,
        standards, date, signature placeholder.
        """
        cl = self.checklist()
        met = [r for r in cl if r.status == "met"]
        return {
            "title": "EU Declaration of Conformity",
            "regulation": "Regulation (EU) 2024/1689 (AI Act)",
            "manufacturer": {
                "name": "Sovereign PIO Project",
                "address": "[To be completed]",
                "contact": "[To be completed]",
            },
            "ai_system": {
                "name": "Sovereign PIO (SPIO)",
                "version": "2.0.0",
                "type": "General-purpose AI system with safety constraints",
                "unique_id": "SPIO-2024-001",
            },
            "conformity_status": {
                "total_requirements": len(cl),
                "requirements_met": len(met),
                "articles_met": [r.article for r in met],
            },
            "harmonised_standards": [
                "ISO/IEC 42001:2023",
                "IEC 62443",
                "ISO/IEC 27001:2022",
            ],
            "assessment_procedure": self.assessment_route(),
            "date": time.strftime("%Y-%m-%d"),
            "signature": {
                "name": "[Authorised representative]",
                "role": "[Role]",
                "signed": False,
            },
            "declaration_id": f"DOC-SPIO-{time.strftime('%Y%m%d')}",
        }

    # ------------------------------------------------------------------
    # EU database entry (Art. 71)
    # ------------------------------------------------------------------

    def eu_database_entry(self) -> dict[str, Any]:
        """Generate Art. 71 EU database registration data.

        Returns structured JSON matching EUDAMED-style schema.
        """
        return {
            "registration_type": "high_risk_ai_system",
            "provider": {
                "name": "Sovereign PIO Project",
                "country": "EU",
                "registration_number": "[To be assigned]",
            },
            "ai_system": {
                "name": "Sovereign PIO (SPIO)",
                "version": "2.0.0",
                "description": (
                    "Personal Intelligent Operator with deterministic "
                    "safety constraints and 12-dimension architecture."
                ),
                "intended_purpose": (
                    "AI-assisted decision support with mandatory "
                    "human oversight."
                ),
                "risk_class": "high",
                "category": "General-purpose AI system",
            },
            "conformity_assessment": {
                "procedure": "internal_control",
                "annex": "VI",
                "notified_body": "N/A (self-assessment)",
                "certificate": "[Pending]",
            },
            "status": "draft",
            "date": time.strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------
    # Assessment route (Annex VI vs VII)
    # ------------------------------------------------------------------

    def assessment_route(self) -> dict[str, Any]:
        """Determine: self-assessment (Annex VI) vs third-party (Annex VII).

        Based on system classification.  SPIO is not in Annex III
        point 1 (biometric) so self-assessment applies per Art. 43(2).
        """
        return {
            "route": "Annex VI — Internal control",
            "reason": (
                "System is not listed in Annex III point 1 "
                "(biometric identification). Per Art. 43(2), "
                "internal control (Annex VI) is sufficient."
            ),
            "third_party_required": False,
            "notified_body": "Not required",
            "self_assessment_steps": [
                "Verify QMS compliance (Art. 17)",
                "Generate Annex IV technical documentation",
                "Run conformity checklist",
                "Sign declaration of conformity (Art. 47)",
                "Register in EU database (Art. 71)",
                "Affix CE marking (Art. 48)",
            ],
        }

    # ------------------------------------------------------------------
    # Gaps
    # ------------------------------------------------------------------

    def gaps(self) -> list[ConformityRequirement]:
        """Requirements not yet met — action items."""
        return [
            r for r in self.checklist()
            if r.status in ("not_met", "partial")
        ]
