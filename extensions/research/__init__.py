"""
Research Extension - Scientific Research Frameworks

8 Research Orchestrators for mathematical and computational problems:

1. ErdosStrausResearcher   - 4/n = 1/a + 1/b + 1/c conjecture analysis
2. BSDResearcher           - Birch-Swinnerton-Dyer conjecture framework
3. NavierStokesResearcher  - Navier-Stokes existence & smoothness
4. RiemannHypothesisResearcher - Riemann Hypothesis research
5. PvsNPResearcher         - P vs NP complexity analysis
6. HodgeResearcher         - Hodge conjecture framework
7. YangMillsResearcher     - Yang-Mills mass gap analysis
8. BenchmarkOrchestrator   - Multi-agent benchmark suite

Research Framework Features:
- Cycle-by-cycle progression with rigor metrics
- Gap identification and documentation
- Phase-based research (baseline -> refinement -> validation)
- Brahim sequence integration for solution analysis

Reference: GPIA Research Infrastructure
"""

from .research_core import (
    # Constants
    PHI,
    BETA,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,
    HARD_RESIDUES,

    # Base classes
    ResearchCycle,
    ResearchPhase,
    ResearchOrchestrator,

    # Researchers
    ErdosStrausResearcher,
    BSDResearcher,
    NavierStokesResearcher,
    RiemannHypothesisResearcher,
    PvsNPResearcher,
    HodgeResearcher,
    YangMillsResearcher,
    BenchmarkOrchestrator,
)

__all__ = [
    # Constants
    "PHI",
    "BETA",
    "BRAHIM_SEQUENCE",
    "BRAHIM_SUM",
    "HARD_RESIDUES",

    # Base classes
    "ResearchCycle",
    "ResearchPhase",
    "ResearchOrchestrator",

    # Researchers
    "ErdosStrausResearcher",
    "BSDResearcher",
    "NavierStokesResearcher",
    "RiemannHypothesisResearcher",
    "PvsNPResearcher",
    "HodgeResearcher",
    "YangMillsResearcher",
    "BenchmarkOrchestrator",
]
