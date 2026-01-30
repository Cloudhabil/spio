"""
Substrate Gateway - FastAPI server wired to SovereignRuntime.

Endpoints:
    POST /query        Process a prompt through PIO → GPIA → ASIOS → Silicon
    GET  /health       Health check (governor hardware status)
    GET  /status       Full runtime status (all 4 layers + inference router)
    GET  /silicon      Inference router capabilities and dispatch stats

Run:
    python -m server.substrate_gateway
    uvicorn server.substrate_gateway:app --host 0.0.0.0 --port 8009
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("substrate_gateway")

# Global runtime — initialized in lifespan
_runtime = None


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request for substrate query."""
    prompt: str
    session_id: str = "api"
    max_tokens: int = 512
    temperature: float = 0.1
    dimension: int | None = Field(
        default=None,
        description="Force a specific dimension (1-12). None = auto-detect.",
    )


class QueryResponse(BaseModel):
    """Response from substrate query."""
    text: str
    session_id: str
    resonance: float = 0.0
    density: float = 0.0
    safe: bool = True
    silicon: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    dimension: int = 7


class AppExecuteRequest(BaseModel):
    """Request to execute an IIAS app."""
    app_name: str
    query: str = ""


class AppExecuteResponse(BaseModel):
    """Response from IIAS app execution."""
    app_id: int
    app_name: str
    category: str
    dimension: int
    dimension_name: str
    silicon_target: str
    silicon_device: str = ""
    silicon_elapsed_ms: float = 0.0
    response: str = ""
    success: bool = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    gpu_available: bool = False
    gpu_name: str = ""
    healthy: bool = True
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Lifespan — boot runtime on startup, shutdown on exit
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Boot SovereignRuntime on startup."""
    global _runtime

    import os
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from sovereign_pio.runtime import RuntimeConfig, SovereignRuntime

    llm_provider = os.environ.get("SPIO_LLM_PROVIDER", "echo")
    llm_model = os.environ.get("SPIO_LLM_MODEL", "llama3.2")
    llm_host = os.environ.get("SPIO_LLM_HOST", "http://localhost:11434")

    config = RuntimeConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_host=llm_host,
        channel="terminal",
    )

    _runtime = SovereignRuntime(config)
    _runtime.boot()
    _runtime._start_time = time.time()

    logger.info(
        "SubstrateGateway booted: llm=%s, model=%s",
        llm_provider, llm_model,
    )
    yield

    # Shutdown
    await _runtime.shutdown()
    logger.info("SubstrateGateway shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sovereign PIO Substrate Gateway",
    description="SPIO API — PIO + GPIA + ASIOS + InferenceRouter",
    version="1.618.2",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def substrate_query(request: QueryRequest):
    """
    Process a prompt through the full SPIO pipeline.

    1. ASIOS governor health check
    2. Wavelength gate audit + enforcement
    3. Silicon dispatch (GPU/NPU/CPU)
    4. PIO reasoning (LLM when configured)
    """
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not booted")

    start = time.perf_counter()

    try:
        response_text = await _runtime.pio.process(
            session_id=request.session_id,
            user_input=request.prompt,
            channel="api",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency = (time.perf_counter() - start) * 1000

    # Extract middleware data from session context
    session = _runtime.pio.get_session(request.session_id)
    wl = session.context.get("wavelength", {}) if session else {}
    silicon = session.context.get("silicon", {}) if session else {}

    return QueryResponse(
        text=response_text,
        session_id=request.session_id,
        resonance=wl.get("resonance", 0.0),
        density=wl.get("density", 0.0),
        safe=wl.get("safe", True),
        silicon=silicon,
        latency_ms=round(latency, 2),
        dimension=silicon.get("dimension", 7),
    )


@app.post("/app/execute", response_model=AppExecuteResponse)
async def execute_app(request: AppExecuteRequest):
    """
    Execute an IIAS app through real silicon.

    Routes the app's category to a dimension, dispatches to the appropriate
    silicon target (NPU/CPU/GPU), and optionally runs LLM reasoning with
    dimension-aware parameters.
    """
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not booted")
    if _runtime.app_executor is None:
        raise HTTPException(status_code=501, detail="AppExecutor not available")

    try:
        result = await _runtime.app_executor.execute(
            app_name=request.app_name,
            query=request.query,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AppExecuteResponse(**result.to_dict())


@app.get("/app/list")
async def list_apps():
    """List all IIAS apps with their execution targets."""
    if _runtime is None or _runtime.app_executor is None:
        return {"apps": [], "count": 0}
    apps = _runtime.app_executor.list_executable()
    return {"apps": apps, "count": len(apps)}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with real hardware status."""
    if _runtime is None:
        return HealthResponse(
            status="booting",
            version="1.618.2",
            uptime_seconds=0,
            healthy=False,
        )

    health = _runtime.asios.governor.check_health()
    gpu = _runtime.asios.governor.monitor.query_gpu()
    uptime = time.time() - getattr(_runtime, "_start_time", time.time())

    return HealthResponse(
        status="healthy" if health.get("healthy") else "degraded",
        version="1.618.2",
        uptime_seconds=round(uptime, 1),
        gpu_available=gpu.available,
        gpu_name=gpu.name if gpu.available else "",
        healthy=health.get("healthy", False),
        warnings=health.get("warnings", []),
    )


@app.get("/status")
async def full_status():
    """Full runtime status from all 4 layers + inference router."""
    if _runtime is None:
        return {"booted": False}
    return _runtime.status()


@app.get("/silicon")
async def silicon_status():
    """Inference router capabilities and dispatch statistics."""
    if _runtime is None or _runtime.inference_router is None:
        return {"available": False}
    return {
        "available": True,
        **_runtime.inference_router.stats(),
    }


@app.get("/app/test-foundation")
async def test_foundation():
    """
    Execute all 5 foundation apps and return computed results.

    Proves that foundation handlers produce real data from
    DimensionRouter, GenesisController, MirrorBalancer,
    LucasAllocator, and PhiSaturator — not generic stubs.
    """
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not booted")
    if _runtime.app_executor is None:
        raise HTTPException(
            status_code=501, detail="AppExecutor not available",
        )

    import json as _json

    foundation_apps = [
        ("dimension_router", "Route 50MB"),
        ("genesis_controller", "Init"),
        ("mirror_balancer", "Balance 1000"),
        ("lucas_allocator", "Allocate 840"),
        ("phi_saturator", "Bandwidth 16"),
    ]

    results: dict = {}
    for app_name, query in foundation_apps:
        try:
            app_result = await _runtime.app_executor.execute(
                app_name=app_name, query=query,
            )
            # Parse JSON response back to dict for clean output
            try:
                computed = _json.loads(app_result.response)
            except (ValueError, TypeError):
                computed = app_result.response
            results[app_name] = {
                "success": app_result.success,
                "silicon_target": app_result.silicon_target,
                "silicon_device": app_result.silicon_device,
                "silicon_elapsed_ms": round(
                    app_result.silicon_elapsed_ms, 4,
                ),
                "computed": computed,
            }
        except Exception as exc:
            results[app_name] = {
                "success": False,
                "error": str(exc),
            }

    passed = sum(1 for v in results.values() if v.get("success"))
    return {
        "foundation_apps": len(foundation_apps),
        "passed": passed,
        "all_passed": passed == len(foundation_apps),
        "results": results,
    }


@app.get("/api/status")
async def api_status():
    """Simple status for Docker health checks."""
    return {"status": "ok", "booted": _runtime is not None and _runtime._booted}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8009):
    """Run the substrate gateway server."""
    import uvicorn
    uvicorn.run(
        "server.substrate_gateway:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
