"""
Inference Router — Real Silicon Dispatch for IIAS

Routes computation to GPU (CUDA), NPU (OpenVINO), or CPU based on
the 12-dimension model from Brahim's Calculator.

    D1-D4  → NPU  (perception, attention, security, stability)
    D5-D8  → CPU  (compression, harmony, reasoning, prediction)
    D9-D12 → GPU  (creativity, wisdom, integration, unification)

Each dispatch actually runs a matrix operation on the target hardware
and returns timing + provenance data proving which silicon executed.

Usage:
    router = InferenceRouter()
    result = router.dispatch(dimension=9, data=np.random.randn(1, 384))
    print(result.device)  # "cuda:0"
    print(result.elapsed_ms)  # real measured time
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sovereign_pio.constants import (
    DIMENSION_NAMES,
    DIMENSION_SILICON,
    LUCAS_NUMBERS,
)

logger = logging.getLogger("inference_router")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Result from a silicon dispatch."""

    dimension: int
    dimension_name: str
    target_silicon: str          # "NPU", "CPU", "GPU"
    device: str                  # "cuda:0", "npu", "cpu"
    backend: str                 # "torch_cuda", "openvino", "torch_cpu", "numpy"
    input_shape: tuple
    output_shape: tuple
    elapsed_ms: float
    throughput_gbps: float       # estimated data throughput
    lucas_capacity: int          # Lucas number for this dimension
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "dimension_name": self.dimension_name,
            "target_silicon": self.target_silicon,
            "device": self.device,
            "backend": self.backend,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "elapsed_ms": round(self.elapsed_ms, 4),
            "throughput_gbps": round(self.throughput_gbps, 6),
            "lucas_capacity": self.lucas_capacity,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Backend probes
# ---------------------------------------------------------------------------

def _probe_cuda() -> bool:
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _probe_openvino_npu() -> bool:
    """Check if OpenVINO NPU device is available."""
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        return "NPU" in devices
    except (ImportError, Exception):
        return False


def _probe_openvino_cpu() -> bool:
    """Check if OpenVINO CPU device is available (fallback for NPU)."""
    try:
        from openvino import Core
        core = Core()
        return "CPU" in core.available_devices
    except (ImportError, Exception):
        return False


# ---------------------------------------------------------------------------
# Dispatch implementations
# ---------------------------------------------------------------------------

def _dispatch_gpu(data: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, str]:
    """Run matmul on CUDA GPU via torch."""
    import torch

    t_data = torch.from_numpy(data).float().cuda()
    t_weight = torch.from_numpy(weight).float().cuda()

    # Synchronize to ensure accurate timing
    torch.cuda.synchronize()
    result = t_data @ t_weight.T
    torch.cuda.synchronize()

    device_name = torch.cuda.get_device_name(0)
    return result.cpu().numpy(), f"cuda:0 ({device_name})"


def _dispatch_npu(data: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, str]:
    """Run matmul on NPU via OpenVINO."""
    import openvino as ov
    from openvino import Core

    core = Core()

    # Build a simple matmul model in OpenVINO
    param_data = ov.opset13.parameter(ov.Shape(data.shape), ov.Type.f32)
    param_weight = ov.opset13.parameter(ov.Shape(weight.shape), ov.Type.f32)
    matmul = ov.opset13.matmul(param_data, param_weight, False, True)
    model = ov.Model([matmul], [param_data, param_weight], "inference_router")

    # Try NPU first, fall back to CPU
    device = "NPU" if "NPU" in core.available_devices else "CPU"
    compiled = core.compile_model(model, device)
    result = compiled({0: data, 1: weight})
    output = result[compiled.output(0)]

    return np.array(output), f"openvino:{device}"


def _dispatch_cpu(data: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, str]:
    """Run matmul on CPU via torch (if available) or numpy."""
    try:
        import torch
        t_data = torch.from_numpy(data).float()
        t_weight = torch.from_numpy(weight).float()
        result = t_data @ t_weight.T
        return result.numpy(), "cpu (torch)"
    except ImportError:
        result = data @ weight.T
        return result, "cpu (numpy)"


# ---------------------------------------------------------------------------
# InferenceRouter
# ---------------------------------------------------------------------------

class InferenceRouter:
    """
    Routes inference to real silicon based on dimension.

    On init, probes available hardware and selects the best backend
    for each silicon target. Gracefully falls back to CPU if hardware
    is unavailable.
    """

    def __init__(self):
        self._has_cuda = _probe_cuda()
        self._has_npu = _probe_openvino_npu()
        self._has_ov_cpu = _probe_openvino_cpu()

        # Track dispatch counts
        self._dispatch_count: dict[str, int] = {"GPU": 0, "CPU": 0, "NPU": 0}

        logger.info(
            "InferenceRouter: CUDA=%s, NPU=%s, OpenVINO-CPU=%s",
            self._has_cuda, self._has_npu, self._has_ov_cpu,
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """Report available silicon backends."""
        return {
            "cuda_gpu": self._has_cuda,
            "openvino_npu": self._has_npu,
            "openvino_cpu": self._has_ov_cpu,
            "torch_cpu": True,
            "numpy_cpu": True,
        }

    def dispatch(
        self,
        dimension: int,
        data: np.ndarray,
        weight: np.ndarray | None = None,
    ) -> DispatchResult:
        """
        Dispatch inference to silicon based on dimension.

        Args:
            dimension: PIO dimension (1-12)
            data: Input data as numpy array (batch × features)
            weight: Optional weight matrix. If None, a PHI-scaled
                    identity-like matrix is generated.

        Returns:
            DispatchResult with timing, device, and output info.
        """
        dim = max(1, min(12, dimension))
        target = DIMENSION_SILICON[dim]
        lucas = LUCAS_NUMBERS[dim - 1]
        dim_name = DIMENSION_NAMES[dim]

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Generate weight matrix if not provided
        if weight is None:
            n_features = data.shape[1]
            # PHI-scaled projection to Lucas-capacity output
            rng = np.random.RandomState(dim * 107)  # deterministic per dimension
            weight = (rng.randn(lucas, n_features) / np.sqrt(n_features)).astype(
                np.float32
            )

        data = data.astype(np.float32)
        weight = weight.astype(np.float32)
        data_bytes = data.nbytes + weight.nbytes

        # Dispatch to silicon
        start = time.perf_counter()
        output, device = self._execute(target, data, weight)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # Backend name
        backend = self._backend_name(target)

        # Throughput: bytes processed / time
        throughput = (data_bytes / 1e9) / (elapsed / 1000) if elapsed > 0 else 0

        self._dispatch_count[target] += 1

        result = DispatchResult(
            dimension=dim,
            dimension_name=dim_name,
            target_silicon=target,
            device=device,
            backend=backend,
            input_shape=tuple(data.shape),
            output_shape=tuple(output.shape),
            elapsed_ms=elapsed,
            throughput_gbps=throughput,
            lucas_capacity=lucas,
        )

        logger.info(
            "Dispatch D%d(%s) → %s [%s] %.2fms",
            dim, dim_name, target, device, elapsed,
        )

        return result

    def dispatch_all(self, data: np.ndarray) -> list[DispatchResult]:
        """Dispatch to all 12 dimensions and return results."""
        return [self.dispatch(d, data) for d in range(1, 13)]

    def stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "capabilities": self.capabilities,
            "dispatch_count": self._dispatch_count.copy(),
            "total_dispatches": sum(self._dispatch_count.values()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(
        self, target: str, data: np.ndarray, weight: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Execute on target silicon with fallback chain."""
        if target == "GPU":
            if self._has_cuda:
                try:
                    return _dispatch_gpu(data, weight)
                except Exception as exc:
                    logger.warning("GPU dispatch failed: %s", exc)
            # Fallback: CPU
            return _dispatch_cpu(data, weight)

        if target == "NPU":
            if self._has_npu or self._has_ov_cpu:
                try:
                    return _dispatch_npu(data, weight)
                except Exception as exc:
                    logger.warning("NPU dispatch failed: %s", exc)
            # Fallback: CPU
            return _dispatch_cpu(data, weight)

        # CPU target — always available
        return _dispatch_cpu(data, weight)

    def _backend_name(self, target: str) -> str:
        """Get the actual backend name for a target."""
        if target == "GPU" and self._has_cuda:
            return "torch_cuda"
        if target == "NPU" and (self._has_npu or self._has_ov_cpu):
            return "openvino"
        return "torch_cpu"
