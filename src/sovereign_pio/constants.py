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
