"""
GPU-accelerated processing for Brahim's Laws.

Provides CUDA/PyTorch batch processing with NumPy fallback.
"""

from .batch_processor import CUDABatchProcessor

__all__ = ["CUDABatchProcessor"]
