"""
EU AI Act Compliance Layer (Regulation 2024/1689).

Modules:
    record_keeping   -- Art. 12: Tamper-evident audit log
    human_oversight  -- Art. 14: GuardianGate + StopSwitch
    risk_management  -- Art. 9:  Lifecycle risk registry
    transparency     -- Art. 13: Deployer-facing disclosure
    robustness       -- Art. 15: Input validation + model integrity
    incident         -- Art. 73: Incident freeze and reporting
    data_governance  -- Art. 10: BIL provenance + data governance
    annex_iv         -- Art. 11: Annex IV technical documentation
    post_market      -- Art. 72: Post-market monitoring + SLO + drift
    qms              -- Art. 17: Quality management system
    conformity       -- Art. 43: Conformity assessment + CE marking
"""

from sovereign_pio.compliance.annex_iv import AnnexIVGenerator
from sovereign_pio.compliance.conformity import (
    ConformityAssessment,
    ConformityRequirement,
)
from sovereign_pio.compliance.data_governance import (
    BrahimIndustryLabel,
    DataGovernance,
)
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
from sovereign_pio.compliance.post_market import (
    SLO,
    PostMarketMonitor,
)
from sovereign_pio.compliance.qms import QualityManagementSystem
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
    "AnnexIVGenerator",
    "AuditRecord",
    "BrahimIndustryLabel",
    "ComplianceAuditLog",
    "ConformityAssessment",
    "ConformityRequirement",
    "DataGovernance",
    "GuardianDecision",
    "GuardianGate",
    "IncidentFreezer",
    "IncidentReport",
    "InputValidator",
    "ModelIntegrityVerifier",
    "OversightInterface",
    "PostMarketMonitor",
    "QualityManagementSystem",
    "Risk",
    "RiskRegistry",
    "SLO",
    "StopSwitch",
    "TransparencyReport",
]
