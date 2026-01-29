"""
GPU-Accelerated Batch Processor for Brahim's Laws.

Uses PyTorch/CUDA for vectorized computation of Brahim's Laws
across thousands of elliptic curves simultaneously.

Optimized for RTX 4070 SUPER:
- 12GB VRAM
- 7168 CUDA cores
- Tensor cores for mixed precision

Author: Elias Oulad Brahim
"""

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Try to import torch, fall back to CPU-only if unavailable
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from ..models.curve_data import EllipticCurveData, Regime
from ..models.analysis_result import BrahimAnalysisResult, BatchAnalysisResult
from ..core.constants import CONSTANTS

logger = logging.getLogger(__name__)


class CUDABatchProcessor:
    """
    GPU-accelerated batch processor for Brahim's Laws.

    Provides massive speedup for analyzing thousands of curves by
    vectorizing all computations on GPU.

    Falls back to CPU (NumPy) if CUDA is unavailable.

    Example:
        processor = CUDABatchProcessor()
        results = processor.process_batch(curves, batch_size=2000)
        stats = processor.compute_statistics(results)
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize CUDA processor.

        Args:
            device: Device string ('cuda', 'cuda:0', 'cpu', etc.)
                   Auto-detects if None
        """
        self.torch_available = TORCH_AVAILABLE

        if self.torch_available:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = None
            logger.warning("PyTorch not available, using CPU (NumPy) backend")

        self._print_device_info()

        # Cache constants
        self.ALPHA_IMTAU = CONSTANTS.ALPHA_IMTAU
        self.BETA_OMEGA = CONSTANTS.BETA_OMEGA
        self.GAMMA_REY = CONSTANTS.GAMMA_REY
        self.REY_C_LOWER = CONSTANTS.REY_C_LOWER
        self.REY_C_UPPER = CONSTANTS.REY_C_UPPER
        self.C = CONSTANTS.CALIBRATION_C

    def _print_device_info(self):
        """Print device information."""
        if self.torch_available and self.device is not None:
            logger.info(f"Device: {self.device}")
            if self.device.type == 'cuda' and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / 1e9
                logger.info(f"GPU: {gpu_name}")
                logger.info(f"Memory: {memory_gb:.2f} GB")
        else:
            logger.info("Using CPU (NumPy) backend")

    # ==========================================================================
    # MAIN BATCH PROCESSING
    # ==========================================================================

    def process_batch(
        self,
        curves: List[EllipticCurveData],
        batch_size: int = 2000
    ) -> List[BrahimAnalysisResult]:
        """
        Process curves in GPU batches.

        Args:
            curves: List of curves to analyze
            batch_size: Curves per GPU batch (tune for VRAM)

        Returns:
            List of analysis results
        """
        if not curves:
            return []

        results = []
        total = len(curves)

        for i in range(0, total, batch_size):
            batch = curves[i:i + batch_size]
            batch_results = self._process_single_batch(batch)
            results.extend(batch_results)

            # Clear GPU memory between batches
            if self.torch_available and self.device is not None and self.device.type == 'cuda':
                torch.cuda.empty_cache()

            logger.debug(f"Processed {min(i + batch_size, total)}/{total} curves")

        return results

    def _process_single_batch(
        self,
        curves: List[EllipticCurveData]
    ) -> List[BrahimAnalysisResult]:
        """Process single batch on GPU/CPU."""
        if self.torch_available and self.device is not None:
            return self._process_batch_torch(curves)
        else:
            return self._process_batch_numpy(curves)

    # ==========================================================================
    # TORCH (GPU) IMPLEMENTATION
    # ==========================================================================

    def _process_batch_torch(
        self,
        curves: List[EllipticCurveData]
    ) -> List[BrahimAnalysisResult]:
        """PyTorch/CUDA implementation."""
        n = len(curves)

        # Extract data to tensors
        N = torch.tensor(
            [c.conductor for c in curves],
            device=self.device, dtype=torch.float64
        )
        Tam = torch.tensor(
            [c.tamagawa_product for c in curves],
            device=self.device, dtype=torch.float64
        )
        Omega = torch.tensor(
            [c.real_period for c in curves],
            device=self.device, dtype=torch.float64
        )
        im_tau = torch.tensor(
            [c.im_tau for c in curves],
            device=self.device, dtype=torch.float64
        )
        sha_actual = torch.tensor(
            [c.sha_analytic if c.sha_analytic else 1.0 for c in curves],
            device=self.device, dtype=torch.float64
        )

        # Law 2: Reynolds Number (vectorized)
        reynolds = self._compute_reynolds_torch(N, Tam, Omega)

        # Law 1: Sha predictions (vectorized)
        sha_imtau = self._predict_sha_imtau_torch(im_tau)
        sha_omega = self._predict_sha_omega_torch(Omega)

        # Law 4: Dynamic scaling (vectorized)
        sha_max = self._predict_sha_max_torch(reynolds)

        # Law 3: Regime classification (vectorized)
        regimes = self._classify_regimes_torch(reynolds)

        # Compute errors
        law1_errors = self._compute_relative_errors_torch(sha_actual, sha_imtau)
        law4_errors = self._compute_relative_errors_torch(sha_actual, sha_max)

        # Move to CPU for result construction
        reynolds_cpu = reynolds.cpu().numpy()
        sha_imtau_cpu = sha_imtau.cpu().numpy()
        sha_omega_cpu = sha_omega.cpu().numpy()
        sha_max_cpu = sha_max.cpu().numpy()
        regimes_cpu = regimes.cpu().numpy()
        law1_errors_cpu = law1_errors.cpu().numpy()
        law4_errors_cpu = law4_errors.cpu().numpy()

        # Construct results
        timestamp = datetime.now().isoformat()
        results = []

        for i, curve in enumerate(curves):
            regime = self._regime_from_code(int(regimes_cpu[i]))

            result = BrahimAnalysisResult(
                curve=curve,
                sha_median_predicted=float(sha_imtau_cpu[i]),
                sha_omega_predicted=float(sha_omega_cpu[i]),
                law1_error=float(law1_errors_cpu[i]),
                reynolds_number=float(reynolds_cpu[i]),
                regime=regime,
                rey_c_lower=self.REY_C_LOWER,
                rey_c_upper=self.REY_C_UPPER,
                sha_max_predicted=float(sha_max_cpu[i]),
                law4_error=float(law4_errors_cpu[i]),
                log_sha_variance=None,
                p_scaling_exponent=-0.25,  # Theoretical value
                consistency_check=0.0,
                is_consistent=True,
                vnand_hash="",
                timestamp=timestamp
            )
            results.append(result)

        return results

    def _compute_reynolds_torch(
        self,
        N: 'torch.Tensor',
        Tam: 'torch.Tensor',
        Omega: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """Vectorized Reynolds computation (PyTorch)."""
        denominator = Tam * Omega
        denominator = torch.where(
            denominator == 0,
            torch.tensor(float('inf'), device=self.device, dtype=torch.float64),
            denominator
        )
        return N / denominator

    def _predict_sha_imtau_torch(self, im_tau: 'torch.Tensor') -> 'torch.Tensor':
        """Vectorized Sha prediction from Im(tau)."""
        return self.C * torch.pow(torch.clamp(im_tau, min=1e-10), self.ALPHA_IMTAU)

    def _predict_sha_omega_torch(self, omega: 'torch.Tensor') -> 'torch.Tensor':
        """Vectorized Sha prediction from Omega."""
        omega_safe = torch.clamp(omega, min=1e-10)
        return self.C * torch.pow(omega_safe, self.BETA_OMEGA)

    def _predict_sha_max_torch(self, reynolds: 'torch.Tensor') -> 'torch.Tensor':
        """Vectorized Sha_max prediction."""
        reynolds_safe = torch.clamp(reynolds, min=1e-10)
        return torch.pow(reynolds_safe, self.GAMMA_REY)

    def _classify_regimes_torch(self, reynolds: 'torch.Tensor') -> 'torch.Tensor':
        """Vectorized regime classification: 0=laminar, 1=transition, 2=turbulent."""
        regimes = torch.ones_like(reynolds, dtype=torch.int32)
        regimes[reynolds < self.REY_C_LOWER] = 0
        regimes[reynolds > self.REY_C_UPPER] = 2
        return regimes

    def _compute_relative_errors_torch(
        self,
        actual: 'torch.Tensor',
        predicted: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """Compute relative errors."""
        actual_safe = torch.where(
            actual == 0,
            torch.tensor(1e-10, device=self.device, dtype=torch.float64),
            actual
        )
        return torch.abs(actual - predicted) / torch.abs(actual_safe)

    # ==========================================================================
    # NUMPY (CPU) IMPLEMENTATION
    # ==========================================================================

    def _process_batch_numpy(
        self,
        curves: List[EllipticCurveData]
    ) -> List[BrahimAnalysisResult]:
        """NumPy/CPU fallback implementation."""
        n = len(curves)

        # Extract data to arrays
        N = np.array([c.conductor for c in curves], dtype=np.float64)
        Tam = np.array([c.tamagawa_product for c in curves], dtype=np.float64)
        Omega = np.array([c.real_period for c in curves], dtype=np.float64)
        im_tau = np.array([c.im_tau for c in curves], dtype=np.float64)
        sha_actual = np.array(
            [c.sha_analytic if c.sha_analytic else 1.0 for c in curves],
            dtype=np.float64
        )

        # Computations
        denominator = Tam * Omega
        denominator = np.where(denominator == 0, np.inf, denominator)
        reynolds = N / denominator

        sha_imtau = self.C * np.power(np.maximum(im_tau, 1e-10), self.ALPHA_IMTAU)
        sha_omega = self.C * np.power(np.maximum(Omega, 1e-10), self.BETA_OMEGA)
        sha_max = np.power(np.maximum(reynolds, 1e-10), self.GAMMA_REY)

        # Regime classification
        regimes = np.ones(n, dtype=np.int32)
        regimes[reynolds < self.REY_C_LOWER] = 0
        regimes[reynolds > self.REY_C_UPPER] = 2

        # Errors
        sha_safe = np.where(sha_actual == 0, 1e-10, sha_actual)
        law1_errors = np.abs(sha_actual - sha_imtau) / np.abs(sha_safe)
        law4_errors = np.abs(sha_actual - sha_max) / np.abs(sha_safe)

        # Construct results
        timestamp = datetime.now().isoformat()
        results = []

        for i, curve in enumerate(curves):
            result = BrahimAnalysisResult(
                curve=curve,
                sha_median_predicted=float(sha_imtau[i]),
                sha_omega_predicted=float(sha_omega[i]),
                law1_error=float(law1_errors[i]),
                reynolds_number=float(reynolds[i]),
                regime=self._regime_from_code(int(regimes[i])),
                rey_c_lower=self.REY_C_LOWER,
                rey_c_upper=self.REY_C_UPPER,
                sha_max_predicted=float(sha_max[i]),
                law4_error=float(law4_errors[i]),
                log_sha_variance=None,
                p_scaling_exponent=-0.25,
                consistency_check=0.0,
                is_consistent=True,
                vnand_hash="",
                timestamp=timestamp
            )
            results.append(result)

        return results

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _regime_from_code(self, code: int) -> Regime:
        """Convert regime code to Regime enum."""
        if code == 0:
            return Regime.LAMINAR
        elif code == 2:
            return Regime.TURBULENT
        return Regime.TRANSITION

    def compute_statistics(
        self,
        results: List[BrahimAnalysisResult]
    ) -> Dict:
        """Compute aggregate statistics from batch results."""
        if not results:
            return {}

        reynolds_vals = np.array([r.reynolds_number for r in results])
        law1_errors = np.array([r.law1_error for r in results])
        law4_errors = np.array([r.law4_error for r in results])

        # Filter finite values
        finite_mask = np.isfinite(reynolds_vals)
        finite_reynolds = reynolds_vals[finite_mask]

        # Regime counts
        laminar = sum(1 for r in results if r.regime == Regime.LAMINAR)
        transition = sum(1 for r in results if r.regime == Regime.TRANSITION)
        turbulent = sum(1 for r in results if r.regime == Regime.TURBULENT)

        # Sha > 1 count
        sha_nontrivial = sum(1 for r in results if r.curve.has_nontrivial_sha)

        return {
            "total_curves": len(results),
            "device": str(self.device) if self.device else "cpu (numpy)",
            "reynolds": {
                "mean": float(np.mean(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
                "median": float(np.median(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
                "std": float(np.std(finite_reynolds)) if len(finite_reynolds) > 0 else 0,
            },
            "regime_distribution": {
                "laminar": laminar,
                "transition": transition,
                "turbulent": turbulent,
            },
            "law1_error": {
                "mean": float(np.mean(law1_errors)),
                "median": float(np.median(law1_errors)),
                "max": float(np.max(law1_errors)),
            },
            "law4_error": {
                "mean": float(np.mean(law4_errors)),
                "median": float(np.median(law4_errors)),
                "max": float(np.max(law4_errors)),
            },
            "sha_nontrivial": {
                "count": sha_nontrivial,
                "rate": sha_nontrivial / len(results) if results else 0,
            }
        }

    def to_batch_result(
        self,
        results: List[BrahimAnalysisResult]
    ) -> BatchAnalysisResult:
        """Convert results list to BatchAnalysisResult."""
        return BatchAnalysisResult(results=results)

    def __repr__(self) -> str:
        device_str = str(self.device) if self.device else "cpu (numpy)"
        return f"CUDABatchProcessor(device='{device_str}')"
