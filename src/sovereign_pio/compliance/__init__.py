"""
EU AI Act Compliance Layer (Regulation 2024/1689).

Modules:
    record_keeping   -- Art. 12: Tamper-evident audit log
    human_oversight  -- Art. 14: GuardianGate + StopSwitch
    risk_management  -- Art. 9:  Lifecycle risk registry
    transparency     -- Art. 13: Deployer-facing disclosure
    robustness       -- Art. 15: Input validation + model integrity
    incident         -- Art. 73: Incident freeze and reporting
"""

from sovereign_pio.compliance.human_oversight import (
    GuardianDecision,
    GuardianGate,
    OversightInterface,
    StopSwitch,
)
from sovereign_pio.compliance.incident import (
    IncidentFreezer,
    IncidentReport,
)
from sovereign_pio.compliance.record_keeping import (
    AuditRecord,
    ComplianceAuditLog,
)
from sovereign_pio.compliance.risk_management import (
    Risk,
    RiskRegistry,
)
from sovereign_pio.compliance.robustness import (
    InputValidator,
    ModelIntegrityVerifier,
)
from sovereign_pio.compliance.transparency import TransparencyReport

__all__ = [
    "AuditRecord",
    "ComplianceAuditLog",
    "GuardianDecision",
    "GuardianGate",
    "IncidentFreezer",
    "IncidentReport",
    "InputValidator",
    "ModelIntegrityVerifier",
    "OversightInterface",
    "Risk",
    "RiskRegistry",
    "StopSwitch",
    "TransparencyReport",
]
