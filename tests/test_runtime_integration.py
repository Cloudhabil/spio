"""
Integration tests for SovereignRuntime.

All tests use echo mode (no LLM, no external deps).
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sovereign_pio.runtime import SovereignRuntime, RuntimeConfig


@pytest.fixture
def runtime():
    """Create and boot a runtime in echo mode."""
    cfg = RuntimeConfig(llm_provider="echo", channel="terminal")
    rt = SovereignRuntime(cfg)
    rt.boot()
    return rt


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_boot_creates_all_layers(runtime):
    """After boot, all four layers should be non-None."""
    assert runtime.pio is not None, "PIO not created"
    assert runtime.memory is not None, "Memory not created"
    assert runtime.asios is not None, "ASIOS not created"
    assert runtime.gateway is not None, "Gateway not created"
    assert runtime.wavelength_gate is not None, "WavelengthGate not created"
    assert runtime._booted is True


def test_pio_processes_message(runtime):
    """PIO should return a non-empty response in echo mode."""
    response = asyncio.get_event_loop().run_until_complete(
        runtime.pio.process("test-session", "Hello, PIO")
    )
    assert response, "Response should be non-empty"
    assert isinstance(response, str)


def test_echo_mode_fallback(runtime):
    """In echo mode (no LLM), response should contain [PIO]."""
    response = asyncio.get_event_loop().run_until_complete(
        runtime.pio.process("echo-session", "What is 2+2?")
    )
    assert "[PIO]" in response, f"Expected [PIO] in echo fallback, got: {response}"


def test_wavelength_gate_audits_input(runtime):
    """After processing, session context should have wavelength data."""
    asyncio.get_event_loop().run_until_complete(
        runtime.pio.process("wl-session", "Test wavelength audit")
    )
    session = runtime.pio.get_session("wl-session")
    assert session is not None
    wl = session.context.get("wavelength")
    assert wl is not None, "Wavelength data missing from session context"
    assert "density" in wl
    assert "resonance" in wl
    assert "safe" in wl


def test_asios_health_in_context(runtime):
    """After processing, session context should have ASIOS health."""
    asyncio.get_event_loop().run_until_complete(
        runtime.pio.process("asios-session", "Check health")
    )
    session = runtime.pio.get_session("asios-session")
    assert session is not None
    health = session.context.get("asios_health")
    assert health is not None, "ASIOS health missing from session context"


def test_pass_broker_has_providers(runtime):
    """PassBroker should have KNOWLEDGE and CAPABILITY providers."""
    from asios.pass_protocol import NeedType

    broker = runtime.asios.pass_broker
    assert len(broker.providers[NeedType.KNOWLEDGE]) >= 1, "No KNOWLEDGE provider"
    assert len(broker.providers[NeedType.CAPABILITY]) >= 1, "No CAPABILITY provider"


def test_memory_stores_and_retrieves(runtime):
    """Memory store + search round-trip should work."""
    content = "The golden ratio is 1.618"
    runtime.memory.store(
        key="test-1",
        content=content,
        metadata={"source": "test"},
    )
    # Key-based retrieval
    entry = runtime.memory.retrieve("test-1")
    assert entry is not None, "retrieve() returned None"
    assert entry.content == content

    # Search returns results (SimpleEmbedder is hash-based, so use
    # the exact same text to guarantee a cosine-similarity match).
    results = runtime.memory.search(content, top_k=1)
    assert len(results) >= 1, "Search returned no results"
    found, similarity = results[0]
    assert found.key == "test-1"
    assert similarity > 0.99, f"Expected near-perfect similarity, got {similarity}"


def test_status_returns_all_layers(runtime):
    """status() should return keys for all four layers."""
    st = runtime.status()
    assert st["booted"] is True
    assert "pio" in st
    assert "gpia" in st
    assert "asios" in st
    assert "moltbot" in st
    assert "wavelength_gate" in st


def test_gateway_has_channel(runtime):
    """Gateway should have at least one registered channel."""
    channels = runtime.gateway.list_channels()
    assert len(channels) >= 1, "No channels registered"
