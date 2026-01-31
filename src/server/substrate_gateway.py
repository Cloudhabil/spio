"""
Substrate Gateway - FastAPI server wired to SovereignRuntime.

Endpoints:
    POST /query        Process a prompt through PIO → GPIA → ASIOS → Silicon
    GET  /health       Health check (governor hardware status)
    GET  /status       Full runtime status (all 4 layers + inference router)
    GET  /silicon      Inference router capabilities and dispatch stats
    GET  /compliance/* EU AI Act deployer API (Art. 26)

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
# Compliance request models (Art. 26 — deployer API)
# ---------------------------------------------------------------------------

class AuditExportRequest(BaseModel):
    """Request to export audit records in a time range."""
    start_timestamp: float
    end_timestamp: float
    event_type: str | None = None


class BILRequest(BaseModel):
    """Request to encode a BIL string."""
    sector: int
    data_type: int
    source: int
    item_id: str


class RiskUpdateRequest(BaseModel):
    """Request to update a risk entry."""
    risk_id: str
    status: str | None = None
    mitigation: str | None = None


class OverrideRequest(BaseModel):
    """Request for a human override."""
    decision_id: str
    human_id: str
    new_decision: str
    reason: str


class HaltRequest(BaseModel):
    """Request for emergency halt."""
    reason: str
    actor: str


class ResumeRequest(BaseModel):
    """Request to resume from halt."""
    reason: str
    actor: str


class BILDecodeRequest(BaseModel):
    """Request to decode a BIL string."""
    bil_string: str


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
# Compliance endpoints (Art. 26 — deployer self-service)
# ---------------------------------------------------------------------------

def _compliance_guard() -> None:
    """Raise 503 if runtime or compliance layer is unavailable."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Runtime not booted")
    if not hasattr(_runtime, "audit_log") or _runtime.audit_log is None:
        raise HTTPException(
            status_code=503, detail="Compliance layer not initialised",
        )


@app.get("/compliance/status")
async def compliance_status():
    """Audit stats + risk posture + halt state."""
    _compliance_guard()
    assert _runtime is not None
    audit_stats = _runtime.audit_log.stats()
    risk_posture = _runtime.risk_registry.assess()
    halted = _runtime.stop_switch.is_halted()
    return {
        "audit": audit_stats,
        "risk_posture": risk_posture,
        "halted": halted,
    }


@app.get("/compliance/transparency")
async def compliance_transparency():
    """Full Art. 13 transparency report."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.transparency_report.full_report()


@app.post("/compliance/audit/export")
async def compliance_audit_export(request: AuditExportRequest):
    """Export audit records in a time range."""
    _compliance_guard()
    assert _runtime is not None
    records = _runtime.audit_log.export(
        request.start_timestamp, request.end_timestamp,
    )
    if request.event_type:
        records = [
            r for r in records
            if r["event_type"] == request.event_type
        ]
    return {"records": records, "count": len(records)}


@app.get("/compliance/audit/verify")
async def compliance_audit_verify():
    """Verify audit chain integrity."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.audit_log.verify_chain()


@app.get("/compliance/risks")
async def compliance_risks():
    """All open risks."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.risk_registry.assess()


@app.post("/compliance/risks/{risk_id}")
async def compliance_risk_update(
    risk_id: str, request: RiskUpdateRequest,
):
    """Update a risk entry."""
    _compliance_guard()
    assert _runtime is not None
    changes: dict[str, Any] = {}
    if request.status is not None:
        changes["status"] = request.status
    if request.mitigation is not None:
        changes["mitigation"] = request.mitigation
    if not changes:
        raise HTTPException(
            status_code=400, detail="No fields to update",
        )
    return _runtime.risk_registry.update(risk_id, **changes)


@app.get("/compliance/incidents")
async def compliance_incidents():
    """Open incidents with deadlines."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.incident_report.list_open()


@app.post("/compliance/override")
async def compliance_override(request: OverrideRequest):
    """Human override of a system decision."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.oversight.override(
        decision_id=request.decision_id,
        human_id=request.human_id,
        new_decision=request.new_decision,
        reason=request.reason,
    )


@app.post("/compliance/halt")
async def compliance_halt(request: HaltRequest):
    """Emergency stop (Art. 14)."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.stop_switch.halt(
        reason=request.reason, actor=request.actor,
    )


@app.post("/compliance/resume")
async def compliance_resume(request: ResumeRequest):
    """Resume from halt (Art. 14)."""
    _compliance_guard()
    assert _runtime is not None
    return _runtime.stop_switch.resume(
        actor=request.actor, reason=request.reason,
    )


@app.post("/compliance/bil/encode")
async def compliance_bil_encode(request: BILRequest):
    """Generate a BIL string from components."""
    from sovereign_pio.compliance.data_governance import (
        BrahimIndustryLabel,
    )

    # Compute simple check digit (sum of digits mod 10)
    digits = f"{request.sector}{request.data_type}{request.source}"
    check = str(sum(int(c) for c in digits if c.isdigit()) % 10)
    bil = BrahimIndustryLabel(
        sector=request.sector,
        data_type=request.data_type,
        source=request.source,
        item_id=request.item_id,
        check_digit=check,
    )
    return bil.to_dict()


@app.post("/compliance/bil/decode")
async def compliance_bil_decode(request: BILDecodeRequest):
    """Parse a BIL string back to structured data."""
    from sovereign_pio.compliance.data_governance import (
        BrahimIndustryLabel,
    )

    try:
        bil = BrahimIndustryLabel.decode(request.bil_string)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return bil.to_dict()


@app.get("/compliance/annex-iv")
async def compliance_annex_iv():
    """Full Annex IV technical documentation."""
    _compliance_guard()
    assert _runtime is not None
    if not hasattr(_runtime, "annex_iv") or _runtime.annex_iv is None:
        raise HTTPException(
            status_code=503, detail="AnnexIVGenerator not initialised",
        )
    return _runtime.annex_iv.generate()


@app.get("/compliance/monitoring")
async def compliance_monitoring():
    """Post-market monitoring dashboard."""
    _compliance_guard()
    assert _runtime is not None
    if (
        not hasattr(_runtime, "post_market")
        or _runtime.post_market is None
    ):
        raise HTTPException(
            status_code=503,
            detail="PostMarketMonitor not initialised",
        )
    return _runtime.post_market.dashboard()


@app.get("/compliance/qms/changes")
async def compliance_qms_changes():
    """QMS change log."""
    _compliance_guard()
    assert _runtime is not None
    if not hasattr(_runtime, "qms") or _runtime.qms is None:
        raise HTTPException(
            status_code=503,
            detail="QualityManagementSystem not initialised",
        )
    return _runtime.qms.list_changes()


@app.get("/compliance/conformity")
async def compliance_conformity():
    """Full conformity checklist."""
    _compliance_guard()
    assert _runtime is not None
    if (
        not hasattr(_runtime, "conformity")
        or _runtime.conformity is None
    ):
        raise HTTPException(
            status_code=503,
            detail="ConformityAssessment not initialised",
        )
    cl = _runtime.conformity.checklist()
    return {
        "requirements": [
            {
                "article": r.article,
                "description": r.description,
                "status": r.status,
                "evidence": r.evidence,
                "last_checked": r.last_checked,
            }
            for r in cl
        ],
        "total": len(cl),
        "met": sum(1 for r in cl if r.status == "met"),
    }


@app.get("/compliance/conformity/gaps")
async def compliance_conformity_gaps():
    """Conformity gaps — requirements not yet met."""
    _compliance_guard()
    assert _runtime is not None
    if (
        not hasattr(_runtime, "conformity")
        or _runtime.conformity is None
    ):
        raise HTTPException(
            status_code=503,
            detail="ConformityAssessment not initialised",
        )
    gaps = _runtime.conformity.gaps()
    return {
        "gaps": [
            {
                "article": r.article,
                "description": r.description,
                "status": r.status,
            }
            for r in gaps
        ],
        "count": len(gaps),
    }


@app.get("/compliance/conformity/declaration")
async def compliance_conformity_declaration():
    """Declaration of conformity (Art. 47)."""
    _compliance_guard()
    assert _runtime is not None
    if (
        not hasattr(_runtime, "conformity")
        or _runtime.conformity is None
    ):
        raise HTTPException(
            status_code=503,
            detail="ConformityAssessment not initialised",
        )
    return _runtime.conformity.declaration_of_conformity()


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
