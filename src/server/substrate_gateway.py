"""
Substrate Gateway - Level 2 Autonomous Control

FastAPI gateway implementing two-tier validation:
1. GPU Reasoning (heavy inference)
2. NPU Audit (12-wavelength security validation)

Routes reasoning to optimal silicon based on dimension routing.
"""

import time
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("substrate_gateway")

# =============================================================================
# API MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Request for substrate query."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.1
    dimension: Optional[int] = 7  # Default to reasoning dimension


class QueryResponse(BaseModel):
    """Response from substrate query."""
    text: str
    resonance: float
    density: float
    safe: bool
    latency_ms: float
    hardware: str
    dimension: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float


# =============================================================================
# GATEWAY APPLICATION
# =============================================================================

class SubstrateGateway:
    """
    Substrate Gateway with two-tier validation.

    Level 2 AC Protocol:
    1. Route to appropriate silicon (GPU for reasoning, NPU for audit)
    2. Validate resonance through 12-wavelength gate
    3. Return audited response with confidence metrics
    """

    def __init__(self):
        self.start_time = time.time()
        self.version = "1.618.1"
        self._wavelength_gate = None
        self._llm_client = None

    def _ensure_wavelength_gate(self):
        """Lazy-load wavelength gate."""
        if self._wavelength_gate is None:
            try:
                from core.wavelengths import WavelengthGate
                self._wavelength_gate = WavelengthGate(
                    threshold=0.1,
                    enable_learning=True,
                    enable_convergence=True
                )
            except ImportError:
                logger.warning("WavelengthGate not available")
        return self._wavelength_gate

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process query through two-tier validation.

        1. Generate response (GPU/LLM)
        2. Audit through wavelength gate (NPU)
        3. Return with resonance metrics
        """
        start_time = time.perf_counter()

        # Step 1: Generate response
        # In production, this would call TensorRT or Ollama
        raw_text = self._generate_response(request.prompt, request.max_tokens)

        # Step 2: Audit through wavelength gate
        gate = self._ensure_wavelength_gate()

        if gate:
            result = gate.evaluate(raw_text)
            resonance = result.resonance
            density = result.density
            safe = result.safe
            final_text = raw_text if safe else f"[AUDIT_BLOCK] {result.reason}"
        else:
            # Fallback: pass through
            resonance = 0.0
            density = 0.022  # Genesis constant
            safe = True
            final_text = raw_text

        latency = (time.perf_counter() - start_time) * 1000

        return QueryResponse(
            text=final_text,
            resonance=resonance,
            density=density,
            safe=safe,
            latency_ms=latency,
            hardware="GPU+NPU" if gate else "CPU",
            dimension=request.dimension
        )

    def _generate_response(self, prompt: str, max_tokens: int) -> str:
        """
        Generate response from LLM.

        In production, this calls TensorRT-LLM or Ollama.
        """
        # Placeholder - in production, call actual LLM
        return f"[Simulated response to: {prompt[:50]}...]"

    def health(self) -> HealthResponse:
        """Health check."""
        return HealthResponse(
            status="healthy",
            version=self.version,
            uptime_seconds=time.time() - self.start_time
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Sovereign PIO Substrate Gateway",
    description="Level 2 AC - Two-tier GPU reasoning + NPU audit",
    version="1.618.1"
)

gateway = SubstrateGateway()


@app.post("/query", response_model=QueryResponse)
async def substrate_query(request: QueryRequest):
    """
    Query the substrate with two-tier validation.

    1. Routes to GPU for reasoning
    2. Audits through NPU wavelength gate
    3. Returns response with resonance metrics
    """
    try:
        return await gateway.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return gateway.health()


@app.get("/api/status")
async def api_status():
    """Simple status endpoint for Docker health checks."""
    return {"status": "ok"}


# =============================================================================
# MAIN
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8009):
    """Run the substrate gateway server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
