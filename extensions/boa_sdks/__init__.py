"""
BOA SDKs Extension - Brahim Onion Agent Scientific Computing Suite

Six specialized SDKs using Brahim mathematical foundations:

1. EgyptianFractions - Fair division, resource optimization, secret splitting
2. SATSolver        - Constraint satisfaction, circuit verification
3. FluidDynamics    - Navier-Stokes simulation, CFD computations
4. TitanExplorer    - Planetary science, atmosphere modeling
5. BrahimDebugger   - Code analysis using golden ratio optimization
6. BOAOrchestrator  - Unified API for all SDKs

Brahim Security Layer:
- PHI: 1.6180339887498949 (Golden Ratio)
- BETA: 0.2360679774997897 (Security constant = sqrt(5) - 2)
- SUM: 214 (Mirror pair sum)

All SDKs share the Brahim Onion encryption wrapper for secure operations.
"""

from .boa_core import (
    # Constants
    PHI,
    BETA_SEC,
    ALPHA_W,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,

    # Security
    BrahimSecurityLayer,

    # Egyptian Fractions
    EgyptianSolution,
    EgyptianFractionsSolver,

    # SAT Solver
    SATResult,
    CNFFormula,
    SATSolution,
    DPLLSolver,

    # Fluid Dynamics
    FlowType,
    FlowConditions,
    SimulationResult,
    NavierStokesSolver,

    # Titan Explorer
    TitanProperties,
    TitanObservation,
    TitanAnalyzer,

    # Debugger
    SafetyVerdict,
    DebugResult,
    BrahimEngine,

    # Orchestrator
    BOAOrchestrator,
)

__all__ = [
    # Constants
    "PHI",
    "BETA_SEC",
    "ALPHA_W",
    "BRAHIM_SEQUENCE",
    "BRAHIM_SUM",

    # Security
    "BrahimSecurityLayer",

    # Egyptian Fractions
    "EgyptianSolution",
    "EgyptianFractionsSolver",

    # SAT Solver
    "SATResult",
    "CNFFormula",
    "SATSolution",
    "DPLLSolver",

    # Fluid Dynamics
    "FlowType",
    "FlowConditions",
    "SimulationResult",
    "NavierStokesSolver",

    # Titan Explorer
    "TitanProperties",
    "TitanObservation",
    "TitanAnalyzer",

    # Debugger
    "SafetyVerdict",
    "DebugResult",
    "BrahimEngine",

    # Orchestrator
    "BOAOrchestrator",
]
