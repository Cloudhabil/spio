"""
Sovereign PIO - Example Usage

Demonstrates how to use all components together:
- PIO (Personal Intelligent Operator)
- GPIA (Intelligence & Reasoning with LLM)
- ASIOS (Runtime & Resource Management)
- Moltbot (Multi-Channel Gateway)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sovereign_pio import PHI, D, Energy
from pio import PIOOperator, logging_middleware
from gpia import Memory, ReasoningEngine, ModelConfig
from asios import ASIOSRuntime, Governor
from moltbot import Gateway, TerminalChannel


async def main():
    """Run Sovereign PIO with terminal channel."""

    print("="*60)
    print("  SOVEREIGN PIO - Personal Intelligent Operator")
    print("="*60)
    print()

    # 1. Initialize ASIOS Runtime
    print("[1/5] Initializing ASIOS Runtime...")
    runtime = ASIOSRuntime()
    print(f"      Governor VRAM limit: {runtime.governor.vram_limit_mb:.0f} MB")
    print(f"      Failsafe threshold: {runtime.failsafe.threshold} failures")

    # 2. Initialize GPIA Memory
    print("[2/5] Initializing GPIA Memory...")
    memory = Memory()
    memory.store("system", "I am Sovereign PIO, a Personal Intelligent Operator.")
    print(f"      Memory initialized with {len(memory.entries)} entries")

    # 3. Initialize GPIA Reasoning (Ollama)
    print("[3/5] Initializing GPIA Reasoning Engine...")
    try:
        # Try to connect to Ollama
        config = ModelConfig.ollama(model="llama3.2")
        reasoning = ReasoningEngine(config)
        print(f"      Connected to Ollama ({config.model})")
        has_reasoning = True
    except Exception as e:
        print(f"      Ollama not available: {e}")
        print("      Running without LLM (responses will be echoed)")
        reasoning = None
        has_reasoning = False

    # 4. Initialize PIO
    print("[4/5] Initializing PIO Operator...")
    pio = PIOOperator()
    pio.use(logging_middleware)
    pio.set_memory(memory)
    if reasoning:
        pio.set_reasoning_engine(reasoning)
    print(f"      PIO ready with {len(pio._handlers)} middleware")

    # 5. Initialize Moltbot Gateway
    print("[5/5] Initializing Moltbot Gateway...")
    gateway = Gateway()

    # Create terminal channel
    terminal = TerminalChannel()

    # Wire PIO to terminal
    async def handle_message(message):
        return await pio.process(
            session_id="terminal",
            user_input=message.content,
            channel="terminal",
        )

    terminal.on_message(handle_message)
    gateway.register(terminal, "terminal")

    print()
    print("="*60)
    print("  System Ready!")
    print(f"  PHI = {PHI:.10f}")
    print(f"  Energy conservation: E(x) = {Energy(0.5):.10f} (2π)")
    print("="*60)
    print()

    # Connect and run
    await gateway.connect("terminal")
    await terminal.listen()

    # Cleanup
    if reasoning:
        await reasoning.close()


async def demo_components():
    """Demo individual components without terminal interaction."""

    print("\n=== Sovereign PIO Component Demo ===\n")

    # Calculator
    print("1. Brahim's Calculator:")
    print(f"   PHI = {PHI}")
    print(f"   D(PHI) = {D(PHI)}")
    print(f"   Energy(0.5) = {Energy(0.5):.10f}")

    # Memory
    print("\n2. GPIA Memory:")
    memory = Memory()
    memory.store("greeting", "Hello, I am Sovereign PIO")
    memory.store("purpose", "I help users with intelligent reasoning")
    results = memory.search("hello", top_k=2)
    print(f"   Stored {len(memory.entries)} entries")
    print(f"   Search 'hello': {len(results)} results")

    # Governor
    print("\n3. ASIOS Governor:")
    gov = Governor()
    health = gov.check_health()
    print(f"   Healthy: {health['healthy']}")
    print(f"   VRAM limit: {gov.vram_limit_mb:.0f} MB")

    # PIO
    print("\n4. PIO Operator:")
    pio = PIOOperator()
    pio.set_memory(memory)
    response = await pio.process("test-session", "Hello!")
    print(f"   Response: {response}")

    print("\n=== Demo Complete ===\n")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        asyncio.run(demo_components())
    else:
        asyncio.run(main())
