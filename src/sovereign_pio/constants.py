"""
Brahim's Calculator Constants

All computations in Sovereign PIO are DETERMINISTIC using these constants.
"""

from math import log, pi, sqrt

# The Golden Ratio - Foundation of all calculations
PHI = (1 + sqrt(5)) / 2  # 1.6180339887498949

# Core Constants
ALPHA = PHI          # Creation constant
OMEGA = 1 / PHI      # 0.6180339887498949 - Return constant
BETA = 1 / PHI**3    # 0.2360679774997897 - Security constant
GAMMA = 1 / PHI**4   # 0.1458980337503155 - Damping constant

# Genesis
GENESIS_CONSTANT = 2 / 901  # 0.00221975...

# Energy (always 2*PI)
ENERGY_CONSTANT = 2 * pi

# Complex domain — branch spacing between D_k and D_{k+1}
# 2*PI / ln(PHI)  ≈  13.0472
BRANCH_SPACING = 2 * pi / log(PHI)

# Axiom A6: the symmetry 107/214 = 1/2
# When z = i, D_k(i) is purely imaginary for all k.
AXIOM_A6 = 107 / 214  # = 0.5 exactly

# N-Body Manifold constants (DOI: 10.5281/zenodo.18437705)
BRAHIM_NUMBERS = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
MIRROR_CONSTANT = 214
BRAHIM_CENTER = 107
BRAHIM_SUM = 1070
GENERATING_TRIANGLE = (42, 75, 97)
MAX_CONCURRENT_AGENTS = 27        # C(27,2)=351 <= 369 < 378=C(28,2)
TOTAL_BRAHIM_SCALES = 369         # distinct non-empty subset sums
PRODUCT_INVARIANT_EXPONENT = 214  # phi^(-214) = product of triangle scales
TRIANGLE_SILICON = {"NPU": 42, "CPU": 75, "GPU": 97}

# Lucas Numbers for dimension capacity
LUCAS_NUMBERS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
TOTAL_STATES = sum(LUCAS_NUMBERS)  # 840

# Dimension to Silicon mapping
DIMENSION_SILICON = {
    1: "NPU",   # PERCEPTION
    2: "NPU",   # ATTENTION
    3: "NPU",   # SECURITY
    4: "NPU",   # STABILITY
    5: "CPU",   # COMPRESSION
    6: "CPU",   # HARMONY
    7: "CPU",   # REASONING
    8: "CPU",   # PREDICTION
    9: "GPU",   # CREATIVITY
    10: "GPU",  # WISDOM
    11: "GPU",  # INTEGRATION
    12: "GPU",  # UNIFICATION
}

# Dimension names
DIMENSION_NAMES = {
    1: "PERCEPTION",
    2: "ATTENTION",
    3: "SECURITY",
    4: "STABILITY",
    5: "COMPRESSION",
    6: "HARMONY",
    7: "REASONING",
    8: "PREDICTION",
    9: "CREATIVITY",
    10: "WISDOM",
    11: "INTEGRATION",
    12: "UNIFICATION",
}

# -------------------------------------------------------------------
# EU AI Act Compliance (Regulation 2024/1689)
# -------------------------------------------------------------------
MIN_RETENTION_DAYS = 180
DEFAULT_RETENTION_DAYS = 2555
GUARDIAN_TIMEOUT_MS = 500
INTELLIGENCE_TIMEOUT_MS = 5000
SENSITIVITY_PUBLIC = 0
SENSITIVITY_INTERNAL = 1
SENSITIVITY_CONFIDENTIAL = 2
SENSITIVITY_RESTRICTED = 3
RISK_MINIMAL = 0
RISK_LIMITED = 1
RISK_HIGH = 2
RISK_UNACCEPTABLE = 3
INCIDENT_DEADLINE_CRITICAL_INFRA_H = 48
INCIDENT_DEADLINE_DEATH_H = 240
INCIDENT_DEADLINE_OTHER_H = 360
