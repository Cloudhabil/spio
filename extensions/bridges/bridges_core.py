"""
Bridges Core - AI Inference Backend Routing

Ported from: CLI-main/src/core/pio_sidecar_bridge.py

Routes PIO 12-dimensions to optimal inference backends:
- D1-D6 (fast): OpenVINO NPU (embeddings, routing)
- D7-D10 (reasoning): Ollama GPU (inference)
- D9-D12 (deep): Sidecar TRT-LLM (heavy inference)
"""

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2
BETA = 1 / PHI**3
LUCAS = {1: 1, 2: 3, 3: 4, 4: 7, 5: 11, 6: 18, 7: 29, 8: 47, 9: 76, 10: 123, 11: 199, 12: 322}

DIMENSION_NAMES = {
    1: "PERCEPTION", 2: "ATTENTION", 3: "SECURITY", 4: "STABILITY",
    5: "COMPRESSION", 6: "HARMONY", 7: "REASONING", 8: "PREDICTION",
    9: "CREATIVITY", 10: "WISDOM", 11: "INTEGRATION", 12: "UNIFICATION",
}


# =============================================================================
# ENUMS
# =============================================================================

class BackendType(Enum):
    """Inference backend types."""
    SIDECAR = "sidecar"      # TensorRT-LLM
    OLLAMA = "ollama"        # Ollama GPU
    OPENVINO = "openvino"    # OpenVINO NPU
    CPU = "cpu"              # CPU fallback


class BackendState(Enum):
    """Backend connection state."""
    UNKNOWN = auto()
    OFFLINE = auto()
    STARTING = auto()
    ONLINE = auto()
    ERROR = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BackendConfig:
    """Backend configuration."""
    name: str
    url: str
    device: str
    dimensions: List[int]
    port: int = 0
    model: str = ""


@dataclass
class QueryResult:
    """Result of a backend query."""
    dimension: int
    dimension_name: str
    capacity: int
    backend: str
    device: str
    response: str
    latency_ms: float
    success: bool = True
    error: str = ""


# =============================================================================
# BACKENDS
# =============================================================================

class BaseBackend:
    """Base class for inference backends."""

    def __init__(self, name: str, device: str, dimensions: List[int]):
        self.name = name
        self.device = device
        self.dimensions = dimensions
        self.state = BackendState.UNKNOWN
        self._lock = threading.Lock()

    def is_alive(self) -> bool:
        """Check if backend is responsive."""
        return False

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Query the backend."""
        return "[Not implemented]"


class SidecarBackend(BaseBackend):
    """TensorRT-LLM sidecar backend."""

    def __init__(self, port: int = 8008):
        super().__init__("Sidecar", "GPU-TRT", [9, 10, 11, 12])
        self.url = f"http://127.0.0.1:{port}"
        self.port = port

    def is_alive(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.url}/docs", timeout=1)
            self.state = BackendState.ONLINE
            return req.status == 200
        except Exception:
            self.state = BackendState.OFFLINE
            return False

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_alive():
            return "[Sidecar offline]"

        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.url}/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("text", "")
        except Exception as e:
            return f"[Sidecar error: {e}]"


class OllamaBackend(BaseBackend):
    """Ollama LLM backend."""

    def __init__(self, port: int = 11434, model: str = "mistral:latest"):
        super().__init__("Ollama", "GPU", [7, 8, 9, 10])
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        self.model = model

    def is_alive(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.url}/api/tags", timeout=2)
            self.state = BackendState.ONLINE
            return req.status == 200
        except Exception:
            self.state = BackendState.OFFLINE
            return False

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_alive():
            return "[Ollama offline]"

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1}
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self.url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("response", "")
        except Exception as e:
            return f"[Ollama error: {e}]"


class OpenVINOBackend(BaseBackend):
    """OpenVINO NPU backend for fast embeddings."""

    def __init__(self):
        super().__init__("OpenVINO", "NPU", [1, 2, 3, 4, 5, 6])
        self.core = None
        self._init()

    def _init(self):
        try:
            import openvino as ov
            self.core = ov.Core()
            devices = self.core.available_devices
            if "NPU" in devices:
                self.device = "NPU"
            elif "GPU" in devices:
                self.device = "GPU"
            else:
                self.device = "CPU"
            self.state = BackendState.ONLINE
        except ImportError:
            self.device = "CPU"
            self.state = BackendState.OFFLINE

    def is_alive(self) -> bool:
        return self.core is not None or self.device == "CPU"

    def embed(self, text: str) -> List[float]:
        """Fast hash-based embedding."""
        h = hashlib.sha384(text.encode()).digest()
        return [b / 255.0 * 2 - 1 for b in h]

    def route(self, text: str, options: List[str]) -> str:
        """Route text to best matching option."""
        if not options:
            return ""
        text_emb = self.embed(text)
        best = options[0]
        best_score = -float('inf')
        for opt in options:
            opt_emb = self.embed(opt)
            score = sum(a * b for a, b in zip(text_emb, opt_emb))
            if score > best_score:
                best_score = score
                best = opt
        return best

    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """For OpenVINO, return routing info instead of generation."""
        return f"[D/{self.device}] Routed via {self.name}"


# =============================================================================
# PIO DIMENSION
# =============================================================================

@dataclass
class PIODimension:
    """A single PIO dimension."""
    n: int
    name: str = ""
    capacity: int = 0
    threshold: float = 0.0

    def __post_init__(self):
        self.name = DIMENSION_NAMES.get(self.n, f"D{self.n}")
        self.capacity = LUCAS.get(self.n, 1)
        self.threshold = 1 / PHI ** self.n


# =============================================================================
# PIO SIDECAR BRIDGE
# =============================================================================

class PIOSidecarBridge:
    """
    Bridges PIO with multiple inference backends.

    Routing logic:
    - D1-D6 (fast): OpenVINO NPU (embeddings, routing)
    - D7-D10 (reasoning): Ollama GPU (inference)
    - D9-D12 (deep): Sidecar TRT-LLM (heavy inference)
    """

    def __init__(self):
        # Initialize backends
        self.sidecar = SidecarBackend()
        self.ollama = OllamaBackend()
        self.openvino = OpenVINOBackend()

        # PIO dimensions
        self.dimensions = {n: PIODimension(n) for n in range(1, 13)}

        # Dimension -> Backend mapping
        self.routing: Dict[int, BaseBackend] = {
            1: self.openvino, 2: self.openvino, 3: self.openvino,
            4: self.openvino, 5: self.openvino, 6: self.openvino,
            7: self.ollama, 8: self.ollama, 9: self.ollama,
            10: self.ollama, 11: self.sidecar, 12: self.sidecar,
        }

        self._lock = threading.Lock()

    def detect_backends(self) -> Dict[str, bool]:
        """Check which backends are available."""
        return {
            "Sidecar (TRT-LLM)": self.sidecar.is_alive(),
            "Ollama (GPU)": self.ollama.is_alive(),
            "OpenVINO (NPU)": self.openvino.is_alive(),
        }

    def route_dimension(self, query: str) -> int:
        """Determine which dimension should handle this query."""
        q = query.lower()

        # Security keywords -> D3
        if any(w in q for w in ["threat", "attack", "security", "protect"]):
            return 3

        # Stability keywords -> D4
        if any(w in q for w in ["balance", "stable", "maintain"]):
            return 4

        # Creativity keywords -> D9
        if any(w in q for w in ["create", "imagine", "novel", "design"]):
            return 9

        # Integration keywords -> D11
        if any(w in q for w in ["integrate", "combine", "merge", "connect"]):
            return 11

        # Unification keywords -> D12
        if any(w in q for w in ["unify", "complete", "total", "all"]):
            return 12

        # Default: D7 (reasoning)
        return 7

    def query(self, prompt: str, dimension: int = None) -> QueryResult:
        """
        Route query through PIO to appropriate backend.
        """
        start = time.time()

        # Determine dimension
        if dimension is None:
            dimension = self.route_dimension(prompt)

        dim = self.dimensions[dimension]
        backend = self.routing[dimension]

        # Try primary backend, fallback to Ollama
        if not backend.is_alive() and backend != self.ollama:
            backend = self.ollama

        # Execute
        try:
            response = backend.query(prompt)
            success = True
            error = ""
        except Exception as e:
            response = ""
            success = False
            error = str(e)

        latency = (time.time() - start) * 1000

        return QueryResult(
            dimension=dimension,
            dimension_name=dim.name,
            capacity=dim.capacity,
            backend=backend.name,
            device=backend.device,
            response=response,
            latency_ms=latency,
            success=success,
            error=error,
        )

    def status(self) -> Dict[str, Any]:
        """Get bridge status."""
        backends = self.detect_backends()
        return {
            "backends": backends,
            "dimensions": {
                n: {
                    "name": d.name,
                    "capacity": d.capacity,
                    "backend": self.routing[n].name,
                }
                for n, d in self.dimensions.items()
            },
            "phi": PHI,
            "beta": BETA,
        }

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """Get dimension routing table."""
        return [
            {
                "dimension": n,
                "name": d.name,
                "capacity": d.capacity,
                "backend": self.routing[n].name,
                "device": self.routing[n].device,
            }
            for n, d in self.dimensions.items()
        ]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_bridge: Optional[PIOSidecarBridge] = None


def create_bridge() -> PIOSidecarBridge:
    """Create a new bridge instance."""
    return PIOSidecarBridge()


def get_bridge() -> PIOSidecarBridge:
    """Get global bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = PIOSidecarBridge()
    return _bridge
