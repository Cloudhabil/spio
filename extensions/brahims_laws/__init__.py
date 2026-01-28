"""
Brahim's Laws Extension - Complete Mathematical Framework

25+ Components implementing Brahim's Laws for physics constants derivation,
elliptic curve analysis, and gauge theory connections.

Categories:
1. Constants (7): PHI, OMEGA, ALPHA, BETA, GAMMA, LOG_PHI, BRAHIM_SEQUENCE
2. Core Laws (6): Laws 1-6 exponents and engine
3. Mechanics (5): BrahimState, MirrorOperator, MirrorProduct, Calculator
4. Geometry (5): BrahimManifold, PythagoreanStructure, GaugeCorrespondence
5. Physics (4): PhysicsConstants, RegulatorTheory, AxiomVerification

Reference: DOI 10.5281/zenodo.18348730
Author: Elias Oulad Brahim
"""

from .brahims_laws_core import (
    # ========================================================================
    # 1. CONSTANTS
    # ========================================================================
    PHI,
    OMEGA,
    ALPHA,
    BETA,
    GAMMA,
    LOG_PHI,

    # Sequence constants
    BRAHIM_SEQUENCE,
    BRAHIM_SEQUENCE_ORIGINAL,
    SUM_CONSTANT,
    CENTER,
    DIMENSION,
    B,
    MIRROR_PAIRS,

    # Law exponents
    ALPHA_IMTAU,
    BETA_OMEGA_EXP,
    GAMMA_REY,
    DELTA_CASCADE,
    REY_C_LOWER,
    REY_C_UPPER,

    # Deviations
    DELTA_4,
    DELTA_5,

    # Experimental values
    EXPERIMENTAL,

    # Consolidated constants
    CONSTANTS,
    BrahimConstants,

    # ========================================================================
    # 2. ENUMS
    # ========================================================================
    Regime,
    Axiom,
    MemoryType,

    # ========================================================================
    # 3. BRAHIM MECHANICS
    # ========================================================================
    BrahimState,
    MirrorOperator,
    MirrorProduct,

    # ========================================================================
    # 4. PHYSICS CONSTANTS
    # ========================================================================
    PhysicsConstants,

    # ========================================================================
    # 5. ELLIPTIC CURVE ANALYSIS
    # ========================================================================
    EllipticCurveData,
    BrahimAnalysisResult,
    BrahimLawsEngine,

    # ========================================================================
    # 6. BRAHIM MANIFOLD & GEOMETRY
    # ========================================================================
    BrahimPoint,
    BrahimManifold,

    # ========================================================================
    # 7. PYTHAGOREAN STRUCTURE
    # ========================================================================
    PythagoreanTriple,
    PythagoreanStructure,

    # ========================================================================
    # 8. GAUGE CORRESPONDENCE
    # ========================================================================
    GaugeGroup,
    GaugeCorrespondence,

    # ========================================================================
    # 9. REGULATOR THEORY
    # ========================================================================
    RegulatorTheory,

    # ========================================================================
    # 10. AXIOM VERIFICATION
    # ========================================================================
    AxiomVerification,
    verify_all_axioms,

    # ========================================================================
    # 11. MAIN INTERFACES
    # ========================================================================
    BrahimNumbersCalculator,
    BrahimGeometry,

    # ========================================================================
    # 12. FACTORY FUNCTIONS
    # ========================================================================
    create_brahim_calculator,
    create_brahim_geometry,
    create_laws_engine,
    create_regulator_theory,
)

__all__ = [
    # Constants
    "PHI",
    "OMEGA",
    "ALPHA",
    "BETA",
    "GAMMA",
    "LOG_PHI",
    "BRAHIM_SEQUENCE",
    "BRAHIM_SEQUENCE_ORIGINAL",
    "SUM_CONSTANT",
    "CENTER",
    "DIMENSION",
    "B",
    "MIRROR_PAIRS",
    "ALPHA_IMTAU",
    "BETA_OMEGA_EXP",
    "GAMMA_REY",
    "DELTA_CASCADE",
    "REY_C_LOWER",
    "REY_C_UPPER",
    "DELTA_4",
    "DELTA_5",
    "EXPERIMENTAL",
    "CONSTANTS",
    "BrahimConstants",

    # Enums
    "Regime",
    "Axiom",
    "MemoryType",

    # Mechanics
    "BrahimState",
    "MirrorOperator",
    "MirrorProduct",

    # Physics
    "PhysicsConstants",

    # Analysis
    "EllipticCurveData",
    "BrahimAnalysisResult",
    "BrahimLawsEngine",

    # Geometry
    "BrahimPoint",
    "BrahimManifold",

    # Pythagorean
    "PythagoreanTriple",
    "PythagoreanStructure",

    # Gauge
    "GaugeGroup",
    "GaugeCorrespondence",

    # Regulator
    "RegulatorTheory",

    # Axioms
    "AxiomVerification",
    "verify_all_axioms",

    # Main interfaces
    "BrahimNumbersCalculator",
    "BrahimGeometry",

    # Factories
    "create_brahim_calculator",
    "create_brahim_geometry",
    "create_laws_engine",
    "create_regulator_theory",
]
