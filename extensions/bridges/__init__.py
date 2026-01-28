"""
Bridges Extension - AI Inference Backend Routing

Ports from: CLI-main/src/core/pio_sidecar_bridge.py

Routes PIO 12-dimensions to inference backends:
- D1-D6: OpenVINO (NPU) - Fast embeddings/routing
- D7-D10: Ollama (GPU) - Reasoning
- D9-D12: Sidecar TensorRT-LLM - Deep inference
"""

from .bridges_core import (
    # Constants
    PHI, BETA, LUCAS,

    # Backends
    SidecarBackend, OllamaBackend, OpenVINOBackend,

    # Core
    PIODimension, PIOSidecarBridge,

    # Factory
    create_bridge, get_bridge,
)

__all__ = [
    "PHI", "BETA", "LUCAS",
    "SidecarBackend", "OllamaBackend", "OpenVINOBackend",
    "PIODimension", "PIOSidecarBridge",
    "create_bridge", "get_bridge",
]
