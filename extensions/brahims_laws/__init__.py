"""
Brahim's Laws Extension for Sovereign-PIO.

A comprehensive framework for analyzing elliptic curves using Brahim's 6 Laws,
connecting the Tate-Shafarevich group behavior to fluid dynamics analogies.

The 6 Laws:
    1. Brahim Conjecture: Sha_median ~ Im(tau)^(2/3) ~ Omega^(-4/3)
    2. Arithmetic Reynolds Number: Rey = N/(Tam*Omega)
    3. Phase Transition: Rey_c in [10, 30]
    4. Dynamic Scaling: Sha_max ~ Rey^(5/12)
    5. Cascade Law: Var(log Sha | p) ~ p^(-1/4)
    6. Consistency Relation: 2/3 = 5/12 + 1/4

Usage:
    from extensions.brahims_laws import (
        BrahimLawsEngine,
        BrahimConstants,
        EllipticCurveData,
        Regime,
        ReynoldsAnalyzer,
        CremonaLoader,
        LMFDBClient
    )

    # Create engine and analyze a curve
    engine = BrahimLawsEngine()
    curve = CremonaLoader().load_by_label("11a1")
    result = engine.analyze(curve)
    print(result.summary())

Author: Elias Oulad Brahim
Version: 2026.1.29
LOC: ~19,000 (full implementation)
"""

__version__ = "2026.1.29"
__author__ = "Elias Oulad Brahim"

# Lazy import cache
_lazy_cache = {}

# Core imports (lightweight - always loaded)
from .core.constants import BrahimConstants, CONSTANTS

# Availability flags - handled by __getattr__ for lazy evaluation
# Do NOT define them here or __getattr__ won't be called
_HAS_FLAGS = {}  # Cache for HAS_* flags

# Mapping of lazy names to their import paths
_LAZY_IMPORTS = {
    # Core (lightweight)
    "BrahimLawsEngine": (".core.brahim_laws", "BrahimLawsEngine"),
    "ReynoldsAnalyzer": (".core.reynolds", "ReynoldsAnalyzer"),
    "ReynoldsStatistics": (".core.reynolds", "ReynoldsStatistics"),
    # Models
    "EllipticCurveData": (".models.curve_data", "EllipticCurveData"),
    "Regime": (".models.curve_data", "Regime"),
    "BrahimAnalysisResult": (".models.analysis_result", "BrahimAnalysisResult"),
    "BatchAnalysisResult": (".models.analysis_result", "BatchAnalysisResult"),
    # Data
    "CremonaLoader": (".data.cremona_loader", "CremonaLoader"),
    "LMFDBClient": (".data.lmfdb_client", "LMFDBClient"),
    # ML (optional - heavy)
    "CurveFeatureExtractor": (".ml.feature_extractor", "CurveFeatureExtractor"),
    "CurveFeatures": (".ml.feature_extractor", "CurveFeatures"),
    "ShaPredictor": (".ml.sha_predictor", "ShaPredictor"),
    "ShaDataset": (".ml.sha_predictor", "ShaDataset"),
    # GPU (optional - heavy)
    "CUDABatchProcessor": (".gpu.batch_processor", "CUDABatchProcessor"),
    # Audit
    "VNANDHasher": (".audit.vnand_hasher", "VNANDHasher"),
    # Advanced - Numbers Calculator
    "BrahimNumbersCalculator": (".brahim_numbers_calculator", "BrahimNumbersCalculator"),
    "BrahimState": (".brahim_numbers_calculator", "BrahimState"),
    "MirrorOperator": (".brahim_numbers_calculator", "MirrorOperator"),
    "PhysicsConstants": (".brahim_numbers_calculator", "PhysicsConstants"),
    # Advanced - Geometry
    "BrahimGeometry": (".brahim_geometry", "BrahimGeometry"),
    "BrahimManifold": (".brahim_geometry", "BrahimManifold"),
    "PythagoreanStructure": (".brahim_geometry", "PythagoreanStructure"),
    "GaugeCorrespondence": (".brahim_geometry", "GaugeCorrespondence"),
    "RegulatorTheory": (".brahim_geometry", "RegulatorTheory"),
    # Agents SDK
    "BrahimCalculatorAgent": (".agents_sdk", "BrahimCalculatorAgent"),
    "execute_function": (".agents_sdk", "execute_function"),
    "BRAHIM_FUNCTIONS": (".agents_sdk", "BRAHIM_FUNCTIONS"),
    # Agent (optional)
    "create_agent": (".agent.curve_agent", "create_agent"),
    "run_agent": (".agent.curve_agent", "run_agent"),
    "CurveAnalysisAgent": (".agent.curve_agent", "CurveAnalysisAgent"),
    # OpenAI Onion Agent
    "BrahimOnionAgent": (".openai_agent", "BrahimOnionAgent"),
    "BrahimAgentBuilder": (".openai_agent", "BrahimAgentBuilder"),
    "BRAHIM_AGENT_TOOLS": (".openai_agent", "BRAHIM_AGENT_TOOLS"),
    "AgentConfig": (".openai_agent", "AgentConfig"),
    "ModelType": (".openai_agent", "ModelType"),
    "Intent": (".openai_agent", "Intent"),
    "LayerID": (".openai_agent", "LayerID"),
    # Output formatters
    "JSONFormatter": (".output.formatters", "JSONFormatter"),
    "TableFormatter": (".output.formatters", "TableFormatter"),
    "RichFormatter": (".output.formatters", "RichFormatter"),
    "get_formatter": (".output.formatters", "get_formatter"),
    # Mobile SDK
    "BrahimAPIServer": (".mobile.api_server", "BrahimAPIServer"),
    "create_app": (".mobile.api_server", "create_app"),
    "MobileConfig": (".mobile.config", "MobileConfig"),
    "APKConfig": (".mobile.config", "APKConfig"),
    "INDUSTRY_PRESETS": (".mobile.config", "INDUSTRY_PRESETS"),
    # CLI
    "cli_app": (".cli", "app"),
    "cli_main": (".cli", "main"),
    # BOA Wavelength Agent
    "BOAWavelengthAgent": (".boa_wavelength_agent", "BOAWavelengthAgent"),
    "BOAResponse": (".boa_wavelength_agent", "BOAResponse"),
    "BOA_WAVELENGTH_TOOLS": (".boa_wavelength_agent", "BOA_WAVELENGTH_TOOLS"),
}


def __getattr__(name: str):
    """Lazy import handler for heavy dependencies."""
    # Check availability flags using _HAS_FLAGS cache
    if name == "HAS_ML":
        if "HAS_ML" not in _HAS_FLAGS:
            try:
                from .ml import feature_extractor
                _HAS_FLAGS["HAS_ML"] = True
            except ImportError:
                _HAS_FLAGS["HAS_ML"] = False
        return _HAS_FLAGS["HAS_ML"]

    if name == "HAS_GPU":
        if "HAS_GPU" not in _HAS_FLAGS:
            try:
                from .gpu import batch_processor
                _HAS_FLAGS["HAS_GPU"] = True
            except ImportError:
                _HAS_FLAGS["HAS_GPU"] = False
        return _HAS_FLAGS["HAS_GPU"]

    if name == "HAS_AGENTS":
        if "HAS_AGENTS" not in _HAS_FLAGS:
            try:
                from .agent import curve_agent
                _HAS_FLAGS["HAS_AGENTS"] = True
            except ImportError:
                _HAS_FLAGS["HAS_AGENTS"] = False
        return _HAS_FLAGS["HAS_AGENTS"]

    if name == "HAS_ONION_AGENT":
        if "HAS_ONION_AGENT" not in _HAS_FLAGS:
            try:
                from . import openai_agent
                _HAS_FLAGS["HAS_ONION_AGENT"] = True
            except ImportError:
                _HAS_FLAGS["HAS_ONION_AGENT"] = False
        return _HAS_FLAGS["HAS_ONION_AGENT"]

    if name == "HAS_FORMATTERS":
        if "HAS_FORMATTERS" not in _HAS_FLAGS:
            try:
                from .output import formatters
                _HAS_FLAGS["HAS_FORMATTERS"] = True
            except ImportError:
                _HAS_FLAGS["HAS_FORMATTERS"] = False
        return _HAS_FLAGS["HAS_FORMATTERS"]

    if name == "HAS_MOBILE":
        if "HAS_MOBILE" not in _HAS_FLAGS:
            try:
                from .mobile import api_server
                _HAS_FLAGS["HAS_MOBILE"] = True
            except ImportError:
                _HAS_FLAGS["HAS_MOBILE"] = False
        return _HAS_FLAGS["HAS_MOBILE"]

    if name == "HAS_CLI":
        if "HAS_CLI" not in _HAS_FLAGS:
            try:
                from . import cli
                _HAS_FLAGS["HAS_CLI"] = True
            except ImportError:
                _HAS_FLAGS["HAS_CLI"] = False
        return _HAS_FLAGS["HAS_CLI"]

    if name == "HAS_BOA_WAVELENGTH":
        if "HAS_BOA_WAVELENGTH" not in _HAS_FLAGS:
            try:
                from . import boa_wavelength_agent
                _HAS_FLAGS["HAS_BOA_WAVELENGTH"] = True
            except ImportError:
                _HAS_FLAGS["HAS_BOA_WAVELENGTH"] = False
        return _HAS_FLAGS["HAS_BOA_WAVELENGTH"]

    # Check cache
    if name in _lazy_cache:
        return _lazy_cache[name]

    # Check if it's a lazy import
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            import importlib
            module = importlib.import_module(module_path, package=__name__)
            value = getattr(module, attr_name)
            _lazy_cache[name] = value
            return value
        except (ImportError, AttributeError) as e:
            # Return None for optional imports
            _lazy_cache[name] = None
            return None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "BrahimConstants",
    "CONSTANTS",
    "BrahimLawsEngine",
    "ReynoldsAnalyzer",
    "ReynoldsStatistics",
    # Models
    "EllipticCurveData",
    "Regime",
    "BrahimAnalysisResult",
    "BatchAnalysisResult",
    # Data
    "CremonaLoader",
    "LMFDBClient",
    # ML (optional)
    "CurveFeatureExtractor",
    "CurveFeatures",
    "ShaPredictor",
    "ShaDataset",
    "HAS_ML",
    # GPU (optional)
    "CUDABatchProcessor",
    "HAS_GPU",
    # Audit
    "VNANDHasher",
    # Advanced - Numbers Calculator
    "BrahimNumbersCalculator",
    "BrahimState",
    "MirrorOperator",
    "PhysicsConstants",
    # Advanced - Geometry
    "BrahimGeometry",
    "BrahimManifold",
    "PythagoreanStructure",
    "GaugeCorrespondence",
    "RegulatorTheory",
    # Advanced - Agents SDK
    "BrahimCalculatorAgent",
    "execute_function",
    "BRAHIM_FUNCTIONS",
    # Agent (optional)
    "create_agent",
    "run_agent",
    "CurveAnalysisAgent",
    "HAS_AGENTS",
    # OpenAI Onion Agent
    "BrahimOnionAgent",
    "BrahimAgentBuilder",
    "BRAHIM_AGENT_TOOLS",
    "AgentConfig",
    "ModelType",
    "Intent",
    "LayerID",
    "HAS_ONION_AGENT",
    # Output formatters
    "JSONFormatter",
    "TableFormatter",
    "RichFormatter",
    "get_formatter",
    "HAS_FORMATTERS",
    # Mobile SDK
    "BrahimAPIServer",
    "create_app",
    "MobileConfig",
    "APKConfig",
    "INDUSTRY_PRESETS",
    "HAS_MOBILE",
    # CLI
    "cli_app",
    "cli_main",
    "HAS_CLI",
    # BOA Wavelength Agent
    "BOAWavelengthAgent",
    "BOAResponse",
    "BOA_WAVELENGTH_TOOLS",
    "HAS_BOA_WAVELENGTH",
]


def get_version() -> str:
    """Return extension version."""
    return __version__


def verify_installation() -> dict:
    """
    Verify all components are importable and functional.

    Returns:
        Dictionary with verification results
    """
    results = {
        "version": __version__,
        "components": {},
        "all_ok": True
    }

    # Test each component (use __getattr__ for lazy loading)
    components = [
        ("BrahimConstants", lambda: CONSTANTS.verify_consistency()),
        ("BrahimLawsEngine", lambda: __getattr__("BrahimLawsEngine")() is not None),
        ("ReynoldsAnalyzer", lambda: __getattr__("ReynoldsAnalyzer")() is not None),
        ("EllipticCurveData", lambda: __getattr__("EllipticCurveData")(label="test") is not None),
        ("CremonaLoader", lambda: __getattr__("CremonaLoader")() is not None),
        ("LMFDBClient", lambda: __getattr__("LMFDBClient")() is not None),
    ]

    for name, test_fn in components:
        try:
            result = test_fn()
            results["components"][name] = {"ok": True, "result": result}
        except Exception as e:
            results["components"][name] = {"ok": False, "error": str(e)}
            results["all_ok"] = False

    return results
