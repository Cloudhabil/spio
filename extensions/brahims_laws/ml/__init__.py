"""
Machine Learning modules for Brahim's Laws extension.

Contains:
    - CurveFeatureExtractor: Transform curves to ML feature vectors
    - ShaPredictor: Predict Sha values using various ML models
    - PhaseClassifier: Classify curves into flow regimes
    - BrahimLawsPINN: Physics-Informed Neural Network
    - RankAwareEmbedder: Learn rank-aware curve embeddings
    - ShaModelTrainer: End-to-end training pipeline

Optional Dependencies:
    - torch: For neural network models
    - sklearn: For GBM, RF models and preprocessing

Uses lazy imports to avoid loading heavy dependencies until needed.
"""

# Lazy import cache
_lazy_cache = {}

# Availability flags (set on first access)
HAS_NEURAL = None
HAS_PINN = None
HAS_CLASSIFIER = None
HAS_EMBEDDINGS = None
HAS_WAVELENGTH = None
HAS_EPHEMERAL = None
HAS_KELIMUTU = None

# Mapping of lazy names to their import paths
_LAZY_IMPORTS = {
    # Core (always available)
    "CurveFeatureExtractor": (".feature_extractor", "CurveFeatureExtractor"),
    "CurveFeatures": (".feature_extractor", "CurveFeatures"),
    "ShaPredictor": (".sha_predictor", "ShaPredictor"),
    "ShaDataset": (".sha_predictor", "ShaDataset"),
    "ShaModelTrainer": (".trainer", "ShaModelTrainer"),
    # Neural (optional)
    "ShaNeuralNet": (".sha_predictor", "ShaNeuralNet"),
    # PINN (optional)
    "BrahimLawsPINN": (".pinn", "BrahimLawsPINN"),
    "PhysicsInformedPredictor": (".pinn", "PhysicsInformedPredictor"),
    # Classifier (optional)
    "PhaseClassifier": (".phase_classifier", "PhaseClassifier"),
    "PhaseClassifierNet": (".phase_classifier", "PhaseClassifierNet"),
    "ClassificationMetrics": (".phase_classifier", "ClassificationMetrics"),
    # Embeddings (optional)
    "RankAwareEmbedder": (".rank_embeddings", "RankAwareEmbedder"),
    "RankAwareEncoder": (".rank_embeddings", "RankAwareEncoder"),
    "CurveDataset": (".rank_embeddings", "CurveDataset"),
    # Wavelength integration
    "WavelengthPipeline": (".wavelength_integration", "WavelengthPipeline"),
    "Intent": (".wavelength_integration", "Intent"),
    "ConvergenceResult": (".wavelength_integration", "ConvergenceResult"),
    "get_pipeline": (".wavelength_integration", "get_pipeline"),
    "process_with_wavelengths": (".wavelength_integration", "process_with_wavelengths"),
    "get_wavelength_stats": (".wavelength_integration", "get_wavelength_stats"),
    "SubstrateState": (".wavelength_integration", "SubstrateState"),
    # Ephemeral subnet (MoE)
    "EphemeralOnionSubnet": (".ephemeral_subnet", "EphemeralOnionSubnet"),
    "BOAEphemeralAgent": (".ephemeral_subnet", "BOAEphemeralAgent"),
    "ExpertSubnet": (".ephemeral_subnet", "ExpertSubnet"),
    "GatingNetwork": (".ephemeral_subnet", "GatingNetwork"),
    "generate_training_data": (".ephemeral_subnet", "generate_training_data"),
    # Kelimutu subnet
    "KelimutuSubnet": (".kelimutu_subnet", "KelimutuSubnet"),
    "BOAKelimutuAgent": (".kelimutu_subnet", "BOAKelimutuAgent"),
    "MagmaSubstrate": (".kelimutu_subnet", "MagmaSubstrate"),
    "UndergroundChannel": (".kelimutu_subnet", "UndergroundChannel"),
    "Lake": (".kelimutu_subnet", "Lake"),
}


def __getattr__(name: str):
    """Lazy import handler for heavy dependencies."""
    global HAS_NEURAL, HAS_PINN, HAS_CLASSIFIER, HAS_EMBEDDINGS
    global HAS_WAVELENGTH, HAS_EPHEMERAL, HAS_KELIMUTU

    # Check availability flags
    if name == "HAS_NEURAL":
        if HAS_NEURAL is None:
            try:
                from .sha_predictor import ShaNeuralNet
                HAS_NEURAL = True
            except ImportError:
                HAS_NEURAL = False
        return HAS_NEURAL

    if name == "HAS_PINN":
        if HAS_PINN is None:
            try:
                from .pinn import BrahimLawsPINN
                HAS_PINN = True
            except ImportError:
                HAS_PINN = False
        return HAS_PINN

    if name == "HAS_CLASSIFIER":
        if HAS_CLASSIFIER is None:
            try:
                from .phase_classifier import PhaseClassifier
                HAS_CLASSIFIER = True
            except ImportError:
                HAS_CLASSIFIER = False
        return HAS_CLASSIFIER

    if name == "HAS_EMBEDDINGS":
        if HAS_EMBEDDINGS is None:
            try:
                from .rank_embeddings import RankAwareEmbedder
                HAS_EMBEDDINGS = True
            except ImportError:
                HAS_EMBEDDINGS = False
        return HAS_EMBEDDINGS

    if name == "HAS_WAVELENGTH":
        if HAS_WAVELENGTH is None:
            try:
                from .wavelength_integration import WavelengthPipeline
                HAS_WAVELENGTH = True
            except ImportError:
                HAS_WAVELENGTH = False
        return HAS_WAVELENGTH

    if name == "HAS_EPHEMERAL":
        if HAS_EPHEMERAL is None:
            try:
                from .ephemeral_subnet import EphemeralOnionSubnet
                HAS_EPHEMERAL = True
            except ImportError:
                HAS_EPHEMERAL = False
        return HAS_EPHEMERAL

    if name == "HAS_KELIMUTU":
        if HAS_KELIMUTU is None:
            try:
                from .kelimutu_subnet import KelimutuSubnet
                HAS_KELIMUTU = True
            except ImportError:
                HAS_KELIMUTU = False
        return HAS_KELIMUTU

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
    # Core exports
    "CurveFeatureExtractor",
    "CurveFeatures",
    "ShaPredictor",
    "ShaDataset",
    "ShaModelTrainer",
    # Neural (optional)
    "ShaNeuralNet",
    "HAS_NEURAL",
    # PINN (optional)
    "BrahimLawsPINN",
    "PhysicsInformedPredictor",
    "HAS_PINN",
    # Classifier (optional)
    "PhaseClassifier",
    "PhaseClassifierNet",
    "ClassificationMetrics",
    "HAS_CLASSIFIER",
    # Embeddings (optional)
    "RankAwareEmbedder",
    "RankAwareEncoder",
    "CurveDataset",
    "HAS_EMBEDDINGS",
    # Wavelength integration
    "WavelengthPipeline",
    "Intent",
    "ConvergenceResult",
    "get_pipeline",
    "process_with_wavelengths",
    "get_wavelength_stats",
    "SubstrateState",
    "HAS_WAVELENGTH",
    # Ephemeral subnet (MoE)
    "EphemeralOnionSubnet",
    "BOAEphemeralAgent",
    "ExpertSubnet",
    "GatingNetwork",
    "generate_training_data",
    "HAS_EPHEMERAL",
    # Kelimutu subnet
    "KelimutuSubnet",
    "BOAKelimutuAgent",
    "MagmaSubstrate",
    "UndergroundChannel",
    "Lake",
    "HAS_KELIMUTU",
]
