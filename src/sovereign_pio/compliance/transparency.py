"""
Art. 13 — Transparency and provision of information to deployers.

Generates deployer-facing disclosure reports: system description,
accuracy metrics, known limitations, and output interpretation
guidelines.
"""

from __future__ import annotations

import time
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog


class TransparencyReport:
    """Exposes system info required by Art. 13 to deployers."""

    SYSTEM_NAME = "Sovereign PIO (SPIO)"
    SYSTEM_VERSION = "1.0.0"

    def __init__(
        self,
        audit_log: ComplianceAuditLog,
        risk_registry: Any = None,
    ) -> None:
        self._audit = audit_log
        self._risk_registry = risk_registry

    def system_description(self) -> dict[str, Any]:
        """Intended purpose, design, versions, hardware."""
        return {
            "name": self.SYSTEM_NAME,
            "version": self.SYSTEM_VERSION,
            "intended_purpose": (
                "Personal Intelligent Operator — AI assistant "
                "with deterministic safety constraints"
            ),
            "design": (
                "Four-layer architecture: PIO (interface), "
                "GPIA (reasoning), ASIOS (governance), "
                "Moltbot (multi-channel gateway)"
            ),
            "hardware_requirements": {
                "minimum_ram_gb": 8,
                "recommended_ram_gb": 32,
                "npu_supported": True,
                "gpu_supported": True,
            },
            "generated_at": time.time(),
        }

    def accuracy_metrics(self) -> dict[str, Any]:
        """Current performance on declared benchmarks."""
        stats = self._audit.stats()
        return {
            "audit_records": stats.get("total_records", 0),
            "chain_integrity": "not_yet_verified",
            "note": (
                "Accuracy metrics depend on deployment context. "
                "Deployers should run domain-specific benchmarks."
            ),
            "generated_at": time.time(),
        }

    def known_limitations(self) -> dict[str, Any]:
        """From RiskRegistry — what the system cannot do safely."""
        limitations: dict[str, Any] = {
            "general": [
                "AI outputs require human review for critical decisions",
                "System does not perform real-time medical diagnosis",
                "System does not perform autonomous legal judgements",
            ],
            "generated_at": time.time(),
        }
        if self._risk_registry is not None:
            try:
                limitations["risk_posture"] = (
                    self._risk_registry.assess()
                )
                limitations["residual_risks"] = (
                    self._risk_registry.residual_report()
                )
            except Exception:
                limitations["risk_posture"] = "unavailable"
        return limitations

    def output_interpretation(self) -> dict[str, Any]:
        """How to read system outputs, confidence scores."""
        return {
            "response_format": (
                "Natural language text with optional structured data"
            ),
            "confidence_scores": (
                "When present, 0.0 = no confidence, 1.0 = full confidence. "
                "Scores below 0.5 should be treated as uncertain."
            ),
            "safety_flags": (
                "Responses may include [WAVELENGTH] or [ASIOS] prefixes "
                "indicating safety-gate interventions."
            ),
            "generated_at": time.time(),
        }

    def full_report(self) -> dict[str, Any]:
        """Complete Art. 13 disclosure package."""
        report = {
            "system_description": self.system_description(),
            "accuracy_metrics": self.accuracy_metrics(),
            "known_limitations": self.known_limitations(),
            "output_interpretation": self.output_interpretation(),
            "regulation": "EU AI Act (Regulation 2024/1689) Art. 13",
            "generated_at": time.time(),
        }
        self._audit.record(
            event_type="transparency_report_generated",
            actor="system",
            detail={"sections": list(report.keys())},
        )
        return report
