"""
EU AI Act operational constants (Regulation 2024/1689).

These are regulatory and engineering constants — not derived from
number theory.  They govern retention, timeouts, risk classification,
and incident deadlines.
"""

# Record-keeping (Art. 12, 19, 26(6))
MIN_RETENTION_DAYS: int = 180           # 6 months minimum
DEFAULT_RETENTION_DAYS: int = 2555      # 7 years (financial sector)

# Timeouts (Art. 14 — human oversight must be possible)
GUARDIAN_TIMEOUT_MS: int = 500          # Max latency for safety gate
INTELLIGENCE_TIMEOUT_MS: int = 5000    # Max latency for reasoning

# Data sensitivity (Art. 10)
SENSITIVITY_PUBLIC: int = 0
SENSITIVITY_INTERNAL: int = 1
SENSITIVITY_CONFIDENTIAL: int = 2
SENSITIVITY_RESTRICTED: int = 3         # PII, health, financial

# Risk levels (Art. 9)
RISK_MINIMAL: int = 0
RISK_LIMITED: int = 1
RISK_HIGH: int = 2
RISK_UNACCEPTABLE: int = 3

# Incident deadlines in hours (Art. 73)
INCIDENT_DEADLINE_CRITICAL_INFRA_H: int = 48   # 2 days
INCIDENT_DEADLINE_DEATH_H: int = 240            # 10 days
INCIDENT_DEADLINE_OTHER_H: int = 360            # 15 days

# Concurrency (validated benchmarks, Jan 2026)
MAX_CONCURRENT_AGENTS: int = 27
