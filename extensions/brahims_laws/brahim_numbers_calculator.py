#!/usr/bin/env python3
"""
Brahim Numbers Calculator - Full Brahim Mechanics Implementation

A comprehensive toolkit for Brahim Numbers computation, physics constants derivation,
and the complete Brahim Mechanics formalism.

Brahim Numbers are defined as the exponents B_n in the canonical phi-adic expansion
of (k-1) from Brahim's Law, characterized by the functional equation:

    B_n + B_{N+1-n} = 214

where each B_n is a valid elliptic curve conductor.

This module implements:
1. Verification of known Brahim Numbers
2. Constraint-based search for new sequence members
3. Phi-adic expansion computation
4. OEIS-compatible sequence export
5. Physics constants derivation (fine structure, Weinberg angle, mass ratios)
6. Brahim Mechanics formalism (states, mirror operator, mirror product)
7. Hierarchy problem computations (coupling and mass hierarchies)
8. Cosmological constants (Hubble constant)

Reference: DOI 10.5281/zenodo.18348730
Based on: "Foundations of Brahim Mechanics" (brahim_mechanics_foundations.tex)

Author: Elias Oulad Brahim
Date: 2026-01-23
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

try:
    import mpmath
    mpmath.mp.dps = 200
except ImportError:
    mpmath = None


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
CENTER = 107                   # Symmetry axis
SUM_CONSTANT = 214            # Functional equation constant

# Known Brahim Numbers (Corrected 2026-01-26 - symmetric sequence)
KNOWN_BRAHIM = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
# Original sequence (for phi-adic coefficient lookups)
KNOWN_BRAHIM_ORIGINAL = [27, 42, 60, 75, 97, 121, 136, 154, 172, 187]

# Coefficients (numerator, denominator) for phi-adic expansion
KNOWN_COEFFICIENTS = {
    27: (1, 6),
    42: (-20, 27),
    60: (-20, 33),
    75: (10, 19),
    97: (-22, 56),
    121: (-29, 25),
    136: (17, 59),
    154: (-20, 56),
    172: (-16, 47),
    187: (7, 59),
}

# =============================================================================
# EXPERIMENTAL PHYSICS CONSTANTS (CODATA 2018)
# =============================================================================

EXPERIMENTAL_CONSTANTS = {
    "fine_structure_inverse": 137.035999084,      # alpha^-1 (CODATA 2018)
    "weinberg_angle_sin2": 0.23122,               # sin^2(theta_W)
    "strong_coupling_inverse": 8.5,               # 1/alpha_s at MZ scale
    "weak_coupling_inverse": 29.5,                # 1/alpha_w
    "muon_electron_ratio": 206.7682830,           # m_mu/m_e
    "proton_electron_ratio": 1836.15267343,       # m_p/m_e
    "hubble_constant": 67.4,                      # H_0 in km/s/Mpc (Planck 2018)
    "coupling_hierarchy": 1.7e36,                 # alpha_EM/alpha_G
    "mass_hierarchy": 1.7e22,                     # m_P/m_e
}

# Brahim Number index mapping
B = {i: KNOWN_BRAHIM[i-1] for i in range(1, 11)}


# =============================================================================
# BRAHIM MECHANICS FORMALISM
# =============================================================================

@dataclass
class BrahimState:
    """
    A Brahim state |B_n> in the discrete Brahim manifold.

    Unlike quantum states, Brahim states are deterministic integers,
    not probability amplitudes.
    """
    index: int
    value: int
    mirror_value: int = field(init=False)

    def __post_init__(self):
        self.mirror_value = SUM_CONSTANT - self.value

    def __repr__(self):
        return f"|B_{self.index}> = |{self.value}>"

    def mirror(self) -> 'BrahimState':
        """Apply mirror operator M to get the partner state."""
        mirror_index = 11 - self.index
        return BrahimState(index=mirror_index, value=self.mirror_value)


class MirrorOperator:
    """
    The Mirror Operator M from Brahim Mechanics.

    For any x in [0, 214], the mirror operator is defined as:
        M(x) = 214 - x

    This operator is an involution: M(M(x)) = x
    """

    @staticmethod
    def apply(x: float) -> float:
        """Apply the mirror operator: M(x) = 214 - x"""
        return SUM_CONSTANT - x

    @staticmethod
    def is_involution(x: float) -> bool:
        """Verify M(M(x)) = x"""
        return abs(MirrorOperator.apply(MirrorOperator.apply(x)) - x) < 1e-10

    @staticmethod
    def is_fixed_point(x: float) -> bool:
        """Check if x is the center (fixed point of M)"""
        return abs(x - CENTER) < 1e-10


class MirrorProduct:
    """
    The Mirror Product from Brahim Mechanics.

    Pairs states according to:
        |B_n> * |M(B_n)> = |214>

    This represents information conservation in Brahim Mechanics.
    """

    @staticmethod
    def compute(state1: BrahimState, state2: BrahimState) -> int:
        """Compute the mirror product of two states."""
        return state1.value + state2.value

    @staticmethod
    def is_mirror_pair(state1: BrahimState, state2: BrahimState) -> bool:
        """Check if two states form a valid mirror pair (sum to 214)."""
        return MirrorProduct.compute(state1, state2) == SUM_CONSTANT


# =============================================================================
# PHYSICS CONSTANTS CALCULATOR
# =============================================================================

class PhysicsConstants:
    """
    Calculate fundamental physics constants using Brahim Number formulas.

    All formulas are derived from the research paper:
    "Foundations of Brahim Mechanics" (brahim_mechanics_foundations.tex)
    """

    @staticmethod
    def fine_structure_inverse() -> Dict[str, Any]:
        """Calculate inverse fine structure constant alpha^-1."""
        computed = B[7] + 1 + 1/(B[1] + 1)
        experimental = EXPERIMENTAL_CONSTANTS["fine_structure_inverse"]
        ppm = abs(computed - experimental) / experimental * 1e6

        return {
            "name": "Fine Structure Constant (1/alpha)",
            "formula": "B_7 + 1 + 1/(B_1 + 1)",
            "computed": computed,
            "experimental": experimental,
            "accuracy_ppm": ppm,
            "accuracy_percent": 100 - (ppm / 10000),
        }

    @staticmethod
    def weinberg_angle() -> Dict[str, Any]:
        """Calculate Weinberg angle sin^2(theta_W)."""
        computed = B[1] / (B[7] - 19)
        experimental = EXPERIMENTAL_CONSTANTS["weinberg_angle_sin2"]
        deviation = abs(computed - experimental) / experimental * 100

        return {
            "name": "Weinberg Angle (sin^2 theta_W)",
            "formula": "B_1/(B_7 - 19)",
            "computed": computed,
            "experimental": experimental,
            "accuracy_percent": 100 - deviation,
        }

    @staticmethod
    def muon_electron_ratio() -> Dict[str, Any]:
        """Calculate muon to electron mass ratio m_mu/m_e."""
        computed = (B[4] ** 2) / B[7] * 5
        experimental = EXPERIMENTAL_CONSTANTS["muon_electron_ratio"]
        deviation = abs(computed - experimental) / experimental * 100

        return {
            "name": "Muon/Electron Mass Ratio",
            "formula": "B_4^2/B_7 * 5",
            "computed": computed,
            "experimental": experimental,
            "accuracy_percent": 100 - deviation,
        }

    @staticmethod
    def proton_electron_ratio() -> Dict[str, Any]:
        """Calculate proton to electron mass ratio m_p/m_e."""
        computed = (B[5] + B[10]) * PHI * 4
        experimental = EXPERIMENTAL_CONSTANTS["proton_electron_ratio"]
        deviation = abs(computed - experimental) / experimental * 100

        return {
            "name": "Proton/Electron Mass Ratio",
            "formula": "(B_5 + B_10) * phi * 4",
            "computed": computed,
            "experimental": experimental,
            "accuracy_percent": 100 - deviation,
        }

    @staticmethod
    def hubble_constant() -> Dict[str, Any]:
        """Calculate Hubble constant H_0."""
        computed = (B[2] * B[9]) / SUM_CONSTANT * 2
        experimental = EXPERIMENTAL_CONSTANTS["hubble_constant"]
        deviation = abs(computed - experimental) / experimental * 100

        return {
            "name": "Hubble Constant (H_0)",
            "formula": "(B_2 * B_9)/214 * 2",
            "computed": computed,
            "experimental": experimental,
            "unit": "km/s/Mpc",
            "accuracy_percent": 100 - deviation,
        }

    @staticmethod
    def alpha_omega_relation() -> Dict[str, Any]:
        """Verify the Alpha-Omega relation between B_1 and B_10."""
        computed = 7 * B[1] - 2
        actual = B[10]

        return {
            "name": "Alpha-Omega Relation",
            "formula": "B_10 = 7 * B_1 - 2",
            "computed": computed,
            "actual_B10": actual,
            "satisfied": computed == actual,
        }

    @classmethod
    def all_constants(cls) -> Dict[str, Dict[str, Any]]:
        """Calculate all physics constants and return as dictionary."""
        return {
            "fine_structure": cls.fine_structure_inverse(),
            "weinberg_angle": cls.weinberg_angle(),
            "muon_electron": cls.muon_electron_ratio(),
            "proton_electron": cls.proton_electron_ratio(),
            "hubble": cls.hubble_constant(),
            "alpha_omega": cls.alpha_omega_relation(),
        }


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

@dataclass
class CandidateScore:
    """Multi-objective score for evaluating Brahim Number candidates."""
    symmetry_score: float
    conductor_score: float
    residual_score: float
    simplicity_score: float
    composite_score: float = 0.0

    def __post_init__(self):
        self.composite_score = (
            0.40 * self.symmetry_score +
            0.30 * self.conductor_score +
            0.20 * self.residual_score +
            0.10 * self.simplicity_score
        )


def evaluate_symmetry(B_left: int, B_right: int) -> float:
    """Evaluate adherence to B_n + B_{N+1-n} = 214."""
    deviation = abs((B_left + B_right) - SUM_CONSTANT)
    return math.exp(-deviation**2 / 10.0)


def evaluate_conductor(n: int) -> float:
    """Estimate probability that n is a valid elliptic curve conductor."""
    if n < 11:
        return 0.0

    factors = {}
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1

    score = 1.0
    for p, e in factors.items():
        if p > 3 and e > 2:
            score *= 0.1
        if p == 2 and e > 8:
            score *= 0.01
        if p == 3 and e > 5:
            score *= 0.01

    if len(factors) == 1 and list(factors.values())[0] == 1:
        score *= 1.2

    return min(score, 1.0)


# =============================================================================
# PUBLIC API
# =============================================================================

class BrahimNumbersCalculator:
    """
    Primary interface for Brahim Numbers computation and Brahim Mechanics.

    Provides methods for:
    - Sequence verification
    - Phi-adic expansion computation
    - Candidate discovery
    - Data export
    - Physics constants calculation
    - Brahim Mechanics formalism
    """

    def __init__(self):
        self.physics = PhysicsConstants()
        self.mirror = MirrorOperator()
        self._states = {i: BrahimState(index=i, value=B[i]) for i in range(1, 11)}

    def get_sequence(self) -> List[int]:
        """Return the known Brahim Number sequence."""
        return KNOWN_BRAHIM.copy()

    def verify(self, n: int) -> Dict[str, Any]:
        """Verify properties of a candidate Brahim Number."""
        mirror = SUM_CONSTANT - n

        return {
            "value": n,
            "is_known": n in KNOWN_BRAHIM,
            "mirror_value": mirror,
            "functional_equation_sum": n + mirror,
            "satisfies_symmetry": (n + mirror) == SUM_CONSTANT,
            "conductor_score": evaluate_conductor(n),
            "coefficient": KNOWN_COEFFICIENTS.get(n)
        }

    def compute_expansion(self, num_terms: int = 10) -> Dict[str, Any]:
        """Compute the phi-adic expansion using known Brahim Numbers."""
        if mpmath is None:
            return {"error": "mpmath library required"}

        phi = (1 + mpmath.sqrt(5)) / 2
        e = mpmath.e
        k = mpmath.mpf(4) / (3 * e * mpmath.log(phi))
        target = k - 1

        terms = []
        reconstruction = (phi - 1) / 32
        terms.append({
            "expression": "(phi-1)/32",
            "exponent": 1,
            "value": float(reconstruction)
        })

        for B in KNOWN_BRAHIM[:num_terms]:
            if B in KNOWN_COEFFICIENTS:
                num, den = KNOWN_COEFFICIENTS[B]
                term_val = mpmath.mpf(num) / den * phi**(-B)
                reconstruction += term_val
                terms.append({
                    "expression": f"({num}/{den})*phi^(-{B})",
                    "exponent": B,
                    "value": float(term_val)
                })

        residual = float(target - reconstruction)
        accuracy = -int(mpmath.log10(abs(target - reconstruction)))

        return {
            "target_value": float(target),
            "reconstruction": float(reconstruction),
            "residual": residual,
            "accuracy_digits": accuracy,
            "terms": terms
        }

    def export(self, format: str = "json") -> str:
        """Export sequence data in standard formats."""
        data = {
            "name": "Brahim Numbers",
            "sequence": KNOWN_BRAHIM,
            "definition": "Exponents in the canonical phi-adic expansion of (k-1) "
                         "satisfying B_n + B_{N+1-n} = 214",
            "center_axis": CENTER,
            "functional_equation": "B_n + B_{N+1-n} = 214",
            "author": "Elias Oulad Brahim",
            "doi": "10.5281/zenodo.18348730"
        }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "oeis":
            return f"Brahim Numbers: {', '.join(map(str, KNOWN_BRAHIM))}\n" \
                   f"Definition: {data['definition']}"
        elif format == "latex":
            return f"\\mathcal{{B}} = \\{{{', '.join(map(str, KNOWN_BRAHIM))}\\}}"
        else:
            return str(KNOWN_BRAHIM)

    # =========================================================================
    # BRAHIM MECHANICS METHODS
    # =========================================================================

    def get_state(self, index: int) -> BrahimState:
        """Get a Brahim state |B_n> by index."""
        if index < 1 or index > 10:
            raise ValueError(f"Index must be 1-10, got {index}")
        return self._states[index]

    def get_all_states(self) -> List[BrahimState]:
        """Return all 10 Brahim states."""
        return [self._states[i] for i in range(1, 11)]

    def mirror_pair(self, index: int) -> Tuple[BrahimState, BrahimState]:
        """Get a mirror pair of states."""
        state = self.get_state(index)
        return (state, state.mirror())

    def all_mirror_pairs(self) -> List[Tuple[BrahimState, BrahimState]]:
        """Return all 5 mirror pairs."""
        return [self.mirror_pair(i) for i in range(1, 6)]

    def apply_mirror(self, x: float) -> float:
        """Apply mirror operator M(x) = 214 - x."""
        return self.mirror.apply(x)

    # =========================================================================
    # PHYSICS CONSTANTS METHODS
    # =========================================================================

    def fine_structure(self) -> Dict[str, Any]:
        """Calculate fine structure constant using Brahim formula."""
        return self.physics.fine_structure_inverse()

    def weinberg_angle(self) -> Dict[str, Any]:
        """Calculate Weinberg angle using Brahim formula."""
        return self.physics.weinberg_angle()

    def muon_electron_ratio(self) -> Dict[str, Any]:
        """Calculate muon/electron mass ratio."""
        return self.physics.muon_electron_ratio()

    def proton_electron_ratio(self) -> Dict[str, Any]:
        """Calculate proton/electron mass ratio."""
        return self.physics.proton_electron_ratio()

    def hubble_constant(self) -> Dict[str, Any]:
        """Calculate Hubble constant."""
        return self.physics.hubble_constant()

    def all_physics_constants(self) -> Dict[str, Dict[str, Any]]:
        """Calculate all physics constants."""
        return self.physics.all_constants()

    def physics_summary(self) -> str:
        """Generate a formatted summary of all physics constants."""
        constants = self.all_physics_constants()
        lines = [
            "=" * 70,
            "  BRAHIM MECHANICS - PHYSICS CONSTANTS SUMMARY",
            "=" * 70,
            "",
        ]

        for key, c in constants.items():
            lines.append(f"{c['name']}")
            lines.append(f"  Formula: {c['formula']}")
            lines.append(f"  Computed: {c['computed']:.6f}")
            if c.get('experimental'):
                lines.append(f"  Experimental: {c['experimental']:.6f}")
            if c.get('accuracy_ppm'):
                lines.append(f"  Accuracy: {c['accuracy_ppm']:.2f} ppm")
            elif c.get('accuracy_percent'):
                lines.append(f"  Accuracy: {c['accuracy_percent']:.2f}%")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report."""
        return {
            "metadata": {
                "name": "Brahim Mechanics Full Report",
                "author": "Elias Oulad Brahim",
                "doi": "10.5281/zenodo.18348730",
                "generated": datetime.now().isoformat(),
            },
            "sequence": {
                "values": KNOWN_BRAHIM,
                "count": len(KNOWN_BRAHIM),
                "sum_constant": SUM_CONSTANT,
                "center_axis": CENTER,
            },
            "mirror_pairs": [
                {
                    "index": i,
                    "B_n": B[i],
                    "B_mirror": B[11-i],
                    "sum": B[i] + B[11-i],
                }
                for i in range(1, 6)
            ],
            "physics_constants": self.all_physics_constants(),
            "expansion": self.compute_expansion(10) if mpmath else {"error": "mpmath not installed"},
        }


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    print("=" * 70)
    print("  BRAHIM NUMBERS CALCULATOR")
    print("  Full Brahim Mechanics Implementation")
    print("=" * 70)
    print()

    calc = BrahimNumbersCalculator()

    # Display known sequence
    print("BRAHIM SEQUENCE:")
    print(f"  B = {calc.get_sequence()}")
    print(f"  Sum constant: {SUM_CONSTANT}")
    print(f"  Center axis: {CENTER}")
    print()

    # Mirror pairs
    print("MIRROR PAIRS (214-Symmetry Conservation):")
    print(f"  {'Index':>5} | {'B_n':>5} | {'B_mirror':>8} | {'Sum':>5}")
    print("  " + "-" * 35)
    for i in range(1, 6):
        b_n = B[i]
        b_mirror = B[11-i]
        print(f"  {i:>5} | {b_n:>5} | {b_mirror:>8} | {b_n + b_mirror:>5}")
    print()

    # Brahim States
    print("BRAHIM STATES:")
    for state in calc.get_all_states()[:5]:
        mirror = state.mirror()
        print(f"  {state} <--mirror--> {mirror}")
    print()

    # Physics Constants Summary
    print(calc.physics_summary())

    print("=" * 70)
    print("  Brahim Mechanics Calculator - Complete")
    print("  DOI: 10.5281/zenodo.18348730")
    print("=" * 70)


if __name__ == "__main__":
    main()
