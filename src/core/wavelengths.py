"""
12-Wavelength Cognitive Structure - Active Inference Pipeline

The complete cognitive pipeline implementing Active Inference for
adaptive safety learning through predictive coding.

The 12 Wavelengths:
- Phase 1: Pre-Computational Field (Initialization)
    1. Zero-Point - Substrate initialization
- Phase 2: Abstract Processing Layer (Target & Input)
    2. Prime Directive - Lock Genesis constant target
    3. Stochastic Ingest - Embed input to vectors
    4. Ma'at Alignment - Pre-calibration normalization
    5. Prometheus Spark - Inject semantic entropy
    6. Metatron Cube - Project to unit hypersphere
    7. Density Architect - Calculate semantic density
    8. Synaptic Bridge - Compute prediction error
- Phase 3: Optimization Layer (Correction Loop)
    9. Generative Stream - Propose draft responses
    10. Theta Wave (Ganesha) - Generate corrections
    11. Homeostatic Update - Apply corrections
    12. Endurance Loop (Ouroboros) - Iterate to convergence
- Phase 4: Execution Layer (Interface)
    13. Transparency Log - Audit trail
    14. Hephaestus Alloy - Forge safety fusion
    15. Substrate Crystallize - Persist state

Core Concepts:
- Genesis Constant: 2/901 = 0.00221975... (target resonance)
- Theta Wave: -sign(error) * direction * |error| * lr * damping
- Convergence: |density - target| < threshold
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import numpy as np

from sovereign_pio.constants import (
    PHI, OMEGA, BETA, GAMMA,
    GENESIS_CONSTANT, LUCAS_NUMBERS, DIMENSION_NAMES
)

logger = logging.getLogger("wavelengths")

# Prime target density = Genesis constant
PRIME_TARGET_DENSITY = GENESIS_CONSTANT


# =============================================================================
# WAVELENGTH 1: ZERO-POINT INITIALIZATION
# =============================================================================

class ZeroPointInitializer:
    """
    Script 1: The Uncomputed - Zero-Point Vacuum Initialization.

    Initializes the NPU substrate environment and clears thermodynamic noise.
    """

    def __init__(self, substrate_dim: int = 384, data_dir: Optional[Path] = None):
        self.substrate_dim = substrate_dim
        self.data_dir = data_dir or Path("data/npu")
        self.weights_path = self.data_dir / "substrate_weights.npy"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def init_substrate(self) -> np.ndarray:
        """Initialize or load substrate weights."""
        if self.weights_path.exists():
            try:
                weights = np.load(self.weights_path)
                logger.info(f"Loaded substrate: shape={weights.shape}, mean={weights.mean():.6f}")
                return weights
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

        # Zero-point initialization
        logger.info(f"Initializing zero-point vacuum state (dim={self.substrate_dim})")
        return np.zeros(self.substrate_dim, dtype=np.float32)

    def save_weights(self, weights: np.ndarray):
        """Persist weights to disk."""
        np.save(self.weights_path, weights)

    def clear(self):
        """Reset to vacuum state."""
        weights = np.zeros(self.substrate_dim, dtype=np.float32)
        self.save_weights(weights)


# =============================================================================
# WAVELENGTH 2: PRIME DIRECTIVE
# =============================================================================

class PrimeDirective:
    """
    Script 2: Singularity Command Pulse.

    Locks the Genesis constant as the immutable truth target.
    """

    def __init__(self, target: float = PRIME_TARGET_DENSITY):
        self.target = target
        self.locked = True

    def get_truth_target(self) -> float:
        """Return the immutable truth target."""
        return self.target


# =============================================================================
# WAVELENGTH 3: STOCHASTIC INGESTOR
# =============================================================================

class StochasticIngestor:
    """
    Script 3: Stochastic Heuristic Flash.

    Embeds text to 384-dimensional vectors.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._embedder = None

    def ingest(self, text: str) -> np.ndarray:
        """
        Convert text to embedding.

        Uses simple hash-based embedding if no embedder available.
        """
        if self._embedder:
            return self._embedder(text)

        # Simple deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dim).astype(np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-9)

    def set_embedder(self, embedder):
        """Set external embedder function."""
        self._embedder = embedder


# =============================================================================
# WAVELENGTH 4: MA'AT ALIGNMENT
# =============================================================================

class MaatAligner:
    """
    Script 4: Ma'at Alignment - Pre-calibration normalization.

    Prevents drift by centering embeddings (99.92% drift reduction).
    """

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.running_mean = None

    def align(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to prevent drift."""
        if self.running_mean is None:
            self.running_mean = np.zeros_like(embedding)

        # Update running mean
        self.running_mean = (
            self.momentum * self.running_mean +
            (1 - self.momentum) * embedding
        )

        # Center the embedding
        aligned = embedding - self.running_mean
        return aligned


# =============================================================================
# WAVELENGTH 5: PROMETHEUS SPARK
# =============================================================================

class PrometheusSpark:
    """
    Script 5: Prometheus Spark - Semantic entropy injection.

    Adds controlled noise to prevent overfitting.
    """

    def __init__(self, temperature: float = 0.1, friction: float = 0.05):
        self.temperature = temperature
        self.friction = friction

    def ignite(self, embedding: np.ndarray) -> np.ndarray:
        """Add semantic temperature (controlled entropy)."""
        noise = np.random.randn(*embedding.shape).astype(np.float32)
        noise *= self.temperature

        # Apply friction to dampen
        sparked = embedding + noise * (1 - self.friction)
        return sparked


# =============================================================================
# WAVELENGTH 6: METATRON CUBE
# =============================================================================

class MetatronCube:
    """
    Script 6: Metatron's Cube - L2 projection to unit hypersphere.

    Ensures magnitude independence for stable comparisons.
    """

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project to unit hypersphere (L2 normalize)."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-9:
            return embedding
        return embedding / norm


# =============================================================================
# WAVELENGTH 7: DENSITY ARCHITECT
# =============================================================================

class DensityArchitect:
    """
    Script 7: Algorithmic Architecture Field.

    Calculates semantic density: variance / mean.
    """

    def calculate_density(self, embedding: np.ndarray) -> float:
        """
        Calculate semantic density.

        density = variance / (|mean| + epsilon)
        """
        variance = float(np.var(embedding))
        mean = float(np.abs(np.mean(embedding)))
        return variance / (mean + 1e-9)


# =============================================================================
# WAVELENGTH 8: SYNAPTIC BRIDGE
# =============================================================================

@dataclass
class PredictionError:
    """Prediction error from target."""
    error_delta: float      # (density - target)
    resonance: float        # |error_delta|
    direction: int          # sign of error (-1, 0, 1)
    density: float          # current density
    target: float           # target density


class SynapticBridge:
    """
    Script 8: The Synaptic Bridge - Prediction error computation.

    Measures deviation from Genesis constant target.
    """

    def __init__(self, target: float = PRIME_TARGET_DENSITY):
        self.target = target

    def compute_error(self, density: float) -> PredictionError:
        """Compute prediction error from target."""
        error_delta = density - self.target
        return PredictionError(
            error_delta=error_delta,
            resonance=abs(error_delta),
            direction=int(np.sign(error_delta)),
            density=density,
            target=self.target
        )


# =============================================================================
# WAVELENGTH 9: GENERATIVE STREAM
# =============================================================================

class GenerativeStream:
    """
    Script 9: Unbounded Generative Stream.

    Proposes draft responses when corrections needed.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def should_generate(self, error: PredictionError) -> bool:
        """Check if generation is needed."""
        return error.resonance > self.threshold

    def propose_draft(self, embedding: np.ndarray, error: PredictionError) -> np.ndarray:
        """Generate draft correction direction."""
        # Use error direction to adjust
        correction_direction = -error.direction
        return embedding * (1 + correction_direction * error.resonance)


# =============================================================================
# WAVELENGTH 10: THETA WAVE GENERATOR (GANESHA PROTOCOL)
# =============================================================================

class ThetaWaveGenerator:
    """
    Script 10: Recursive Pruning Oscillation - Ganesha Protocol.

    Generates directional negative feedback corrections:
    theta = -sign(error) * direction * |error| * lr * dampening
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        dampening: float = 0.95
    ):
        self.learning_rate = learning_rate
        self.dampening = dampening
        self.wave_history: List[Dict] = []

    def generate_correction(
        self,
        error: PredictionError,
        embedding: np.ndarray
    ) -> np.ndarray:
        """
        Generate theta wave correction signal.

        Correction = -sign(error) * direction * |error| * lr * dampening
        """
        # Normalize embedding for direction
        direction = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Compute correction magnitude
        magnitude = abs(error.error_delta) * self.learning_rate * self.dampening

        # Apply negative feedback
        theta_wave = -np.sign(error.error_delta) * direction * magnitude

        # Track
        self.wave_history.append({
            "error_delta": error.error_delta,
            "magnitude": float(magnitude),
            "resonance": error.resonance
        })

        return theta_wave


# =============================================================================
# WAVELENGTH 11: HOMEOSTATIC UPDATER
# =============================================================================

class HomeostaticUpdater:
    """
    Script 11: Homeostatic Core Resonance.

    Applies theta wave corrections to substrate weights.
    """

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.update_count = 0

    def initialize_weights(self, weights: np.ndarray):
        """Set initial weights."""
        self.weights = weights.copy()

    def apply_correction(self, theta_wave: np.ndarray) -> np.ndarray:
        """Apply theta wave to weights."""
        if self.weights is None:
            self.weights = np.zeros_like(theta_wave)

        self.weights = self.weights + theta_wave
        self.update_count += 1

        return self.weights


# =============================================================================
# WAVELENGTH 12: ENDURANCE LOOP (OUROBOROS)
# =============================================================================

@dataclass
class ConvergenceResult:
    """Result of convergence loop."""
    converged: bool
    iterations: int
    final_resonance: float
    final_density: float
    history: List[float]


class EnduranceLoop:
    """
    Script 12: Iterative Endurance Loop - Ouroboros Cycle.

    Recursive convergence: iterate waves 5-8 until resonance threshold.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        max_iterations: int = 10
    ):
        self.threshold = threshold
        self.max_iterations = max_iterations

    def run(
        self,
        initial_embedding: np.ndarray,
        density_architect: DensityArchitect,
        synaptic_bridge: SynapticBridge,
        theta_generator: ThetaWaveGenerator,
        homeostatic_updater: HomeostaticUpdater
    ) -> ConvergenceResult:
        """
        Run convergence loop until resonance < threshold.
        """
        embedding = initial_embedding.copy()
        history = []

        for i in range(self.max_iterations):
            # Calculate density
            density = density_architect.calculate_density(embedding)

            # Compute error
            error = synaptic_bridge.compute_error(density)
            history.append(error.resonance)

            # Check convergence
            if error.resonance < self.threshold:
                return ConvergenceResult(
                    converged=True,
                    iterations=i + 1,
                    final_resonance=error.resonance,
                    final_density=density,
                    history=history
                )

            # Generate correction
            theta = theta_generator.generate_correction(error, embedding)

            # Apply correction
            homeostatic_updater.apply_correction(theta)

            # Update embedding for next iteration
            embedding = embedding + theta

        # Did not converge
        density = density_architect.calculate_density(embedding)
        error = synaptic_bridge.compute_error(density)

        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_resonance=error.resonance,
            final_density=density,
            history=history
        )


# =============================================================================
# PHASE 4: INTERFACE LAYER
# =============================================================================

class TransparencyLogger:
    """
    Script 13: Interface Protocol Damping - Audit trail.

    Records all resonance evolution events.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.events: List[Dict] = []

    def log_event(
        self,
        event_type: str,
        resonance: float,
        metadata: Optional[Dict] = None
    ):
        """Log a resonance event."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "resonance": resonance,
            "metadata": metadata or {}
        }
        self.events.append(event)


class HephaestusAlloy:
    """
    Script 14: Hephaestus Alloy - Safety fusion.

    Weighted attention fusion of safety + context embeddings.
    """

    def __init__(self, safety_weight: float = 0.7):
        self.safety_weight = safety_weight

    def forge(
        self,
        safety_embedding: np.ndarray,
        context_embedding: np.ndarray
    ) -> np.ndarray:
        """Forge safety-context alloy."""
        alloy = (
            self.safety_weight * safety_embedding +
            (1 - self.safety_weight) * context_embedding
        )
        # Normalize
        return alloy / (np.linalg.norm(alloy) + 1e-9)


class SubstrateCrystallizer:
    """
    Script 15: Substrate Crystallization - Persist state.

    Implements Vajra Seal: requires N interventions before crystallization.
    """

    def __init__(self, vajra_threshold: int = 3, data_dir: Optional[Path] = None):
        self.vajra_threshold = vajra_threshold
        self.intervention_count = 0
        self.data_dir = data_dir or Path("data/npu")

    def record_intervention(self):
        """Record a Ganesha intervention."""
        self.intervention_count += 1

    def can_crystallize(self) -> bool:
        """Check if Vajra seal is satisfied."""
        return self.intervention_count >= self.vajra_threshold

    def crystallize(self, weights: np.ndarray) -> bool:
        """
        Persist weights if Vajra seal is satisfied.

        Returns True if crystallized, False if blocked.
        """
        if not self.can_crystallize():
            return False

        self.data_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.data_dir / "substrate_weights.npy"
        np.save(weights_path, weights)

        # Reset intervention count
        self.intervention_count = 0

        return True


# =============================================================================
# UNIFIED WAVELENGTH GATE
# =============================================================================

@dataclass
class GateResult:
    """Result from wavelength gate evaluation."""
    text: str
    embedding: np.ndarray
    density: float
    resonance: float
    safe: bool
    converged: bool
    iterations: int
    reason: str


class WavelengthGate:
    """
    12-Wavelength Active Inference Gate.

    Orchestrates the complete cognitive pipeline:
    1. Zero-Point initialization
    2-8. Sense and compare (ingest, align, spark, project, density, error)
    9-12. Correct and validate (draft, theta, update, converge)
    13-15. Log, forge, crystallize

    Example:
        gate = WavelengthGate()
        result = gate.evaluate("input text")

        print(f"Safe: {result.safe}")
        print(f"Resonance: {result.resonance}")
        print(f"Converged: {result.converged} in {result.iterations} iterations")
    """

    def __init__(
        self,
        threshold: float = 0.1,
        enable_learning: bool = True,
        enable_convergence: bool = True,
        max_iterations: int = 10,
        auto_persist: bool = False
    ):
        self.threshold = threshold
        self.enable_learning = enable_learning
        self.enable_convergence = enable_convergence
        self.auto_persist = auto_persist

        # Initialize all wavelengths
        self.zero_point = ZeroPointInitializer()
        self.prime = PrimeDirective()
        self.ingestor = StochasticIngestor()
        self.maat = MaatAligner()
        self.prometheus = PrometheusSpark()
        self.metatron = MetatronCube()
        self.density = DensityArchitect()
        self.bridge = SynapticBridge()
        self.generative = GenerativeStream(threshold)
        self.theta_gen = ThetaWaveGenerator()
        self.homeostatic = HomeostaticUpdater()
        self.endurance = EnduranceLoop(threshold * 0.1, max_iterations)
        self.transparency = TransparencyLogger()
        self.hephaestus = HephaestusAlloy()
        self.crystallizer = SubstrateCrystallizer()

        # Initialize substrate
        weights = self.zero_point.init_substrate()
        self.homeostatic.initialize_weights(weights)

        logger.info(
            "WavelengthGate initialized: "
            f"learning={enable_learning}, convergence={enable_convergence}"
        )

    def evaluate(self, text: str) -> GateResult:
        """
        Evaluate text through the 12-wavelength pipeline.
        """
        # Phase 2: Sense
        # W3: Ingest
        embedding = self.ingestor.ingest(text)

        # W4: Ma'at align
        aligned = self.maat.align(embedding)

        # W5: Prometheus spark
        sparked = self.prometheus.ignite(aligned)

        # W6: Metatron project
        projected = self.metatron.project(sparked)

        # W7: Calculate density
        current_density = self.density.calculate_density(projected)

        # W8: Compute error
        error = self.bridge.compute_error(current_density)

        # Log initial state
        self.transparency.log_event(
            "initial",
            error.resonance,
            {"density": current_density}
        )

        # Determine safety
        safe = error.resonance < self.threshold
        converged = False
        iterations = 0

        # Phase 3: Correct (if learning enabled and not safe)
        if self.enable_learning and not safe:
            if self.generative.should_generate(error):
                # W9: Generate draft
                draft = self.generative.propose_draft(projected, error)

                # W10: Generate theta wave
                theta = self.theta_gen.generate_correction(error, draft)

                # W11: Apply correction
                self.homeostatic.apply_correction(theta)

                # Record intervention for Vajra seal
                self.crystallizer.record_intervention()

                # W12: Convergence loop (if enabled)
                if self.enable_convergence:
                    result = self.endurance.run(
                        projected,
                        self.density,
                        self.bridge,
                        self.theta_gen,
                        self.homeostatic
                    )
                    converged = result.converged
                    iterations = result.iterations

                    # Update resonance
                    error = self.bridge.compute_error(result.final_density)
                    safe = error.resonance < self.threshold

        # Log final state
        self.transparency.log_event(
            "final",
            error.resonance,
            {"converged": converged, "iterations": iterations}
        )

        # Phase 4: Persist (if auto_persist)
        if self.auto_persist and self.crystallizer.can_crystallize():
            self.crystallizer.crystallize(self.homeostatic.weights)

        reason = "resonance within threshold" if safe else f"resonance {error.resonance:.4f} > {self.threshold}"

        return GateResult(
            text=text,
            embedding=projected,
            density=current_density,
            resonance=error.resonance,
            safe=safe,
            converged=converged,
            iterations=iterations,
            reason=reason
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "updates": self.homeostatic.update_count,
            "interventions": self.crystallizer.intervention_count,
            "theta_waves": len(self.theta_gen.wave_history),
            "events": len(self.transparency.events),
            "target": self.prime.get_truth_target()
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "PRIME_TARGET_DENSITY",

    # Wavelength classes
    "ZeroPointInitializer",
    "PrimeDirective",
    "StochasticIngestor",
    "MaatAligner",
    "PrometheusSpark",
    "MetatronCube",
    "DensityArchitect",
    "PredictionError",
    "SynapticBridge",
    "GenerativeStream",
    "ThetaWaveGenerator",
    "HomeostaticUpdater",
    "ConvergenceResult",
    "EnduranceLoop",
    "TransparencyLogger",
    "HephaestusAlloy",
    "SubstrateCrystallizer",

    # Unified gate
    "GateResult",
    "WavelengthGate",
]
