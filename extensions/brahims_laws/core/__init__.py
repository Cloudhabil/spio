"""
Core computation modules for Brahim's Laws extension.

Contains:
    - BrahimConstants: Mathematical constants and exponents
    - BrahimLawsEngine: Main analysis engine implementing all 6 laws
    - ReynoldsAnalyzer: Specialized Reynolds number analysis
    - Alphabet: Compressed JSON notation system
    - FundamentalAlphabet: Symbolic algebra for exact identities
    - GeneratingFunction: The generating function connecting phi, e, 2/3
    - OptimizedIO: High-performance I/O utilities
"""

from .constants import BrahimConstants, CONSTANTS
from .brahim_laws import BrahimLawsEngine
from .reynolds import ReynoldsAnalyzer, ReynoldsStatistics
from .alphabet import compress, expand, ALPHABET, ALPHABET_REVERSE, AlphabetSchema
from .generating_function import (
    GeneratingFunction,
    MasterIdentity,
    BrahimCalculator,
    FundamentalConstants,
    fibonacci,
    zeckendorf,
)
from .optimized_io import (
    OptimizedWriter,
    OptimizedReader,
    BatchIO,
    OutputFormat,
    IOMetrics,
)

# Optional high-precision alphabet
try:
    from .fundamental_alphabet import (
        Symbol,
        Constant,
        Identity,
        compute_identities,
        CONSTANTS as FUNDAMENTAL_CONSTANTS,
    )
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False

__all__ = [
    # Original exports
    "BrahimConstants",
    "CONSTANTS",
    "BrahimLawsEngine",
    "ReynoldsAnalyzer",
    "ReynoldsStatistics",
    # Alphabet
    "compress",
    "expand",
    "ALPHABET",
    "ALPHABET_REVERSE",
    "AlphabetSchema",
    # Generating function
    "GeneratingFunction",
    "MasterIdentity",
    "BrahimCalculator",
    "FundamentalConstants",
    "fibonacci",
    "zeckendorf",
    # Optimized I/O
    "OptimizedWriter",
    "OptimizedReader",
    "BatchIO",
    "OutputFormat",
    "IOMetrics",
    # Feature flag
    "HAS_MPMATH",
]
