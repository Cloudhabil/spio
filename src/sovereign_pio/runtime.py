"""
Sovereign Runtime - Wires PIO, GPIA, ASIOS, and Moltbot into one system.

Boot sequence:
  1. WavelengthGate  (cognitive pipeline)
  2. Memory + optional ReasoningEngine  (GPIA)
  3. ASIOSRuntime  (governor, government, pass broker)
  4. PIOOperator  (interface + middleware)
  5. Gateway + channel  (Moltbot)
  6. Register PassBroker providers
  7. Wire gateway handler -> PIO.process
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("sovereign_runtime")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Configuration for the Sovereign Runtime."""

    llm_provider: str = "echo"           # "echo", "ollama", "openai"
    llm_model: str = "llama3.2"          # model name for ollama/openai
    llm_host: str = "http://localhost:11434"
    openai_api_key: str = ""
    embedding_model: str = "nomic-embed-text"  # Ollama embedding model

    channel: str = "terminal"            # "terminal", "telegram", "discord"
    channel_token: str = ""              # token for telegram/discord

    wavelength_threshold: float = 0.1
    wavelength_enforce: bool = True      # block requests that fail safety gate
    memory_persist_path: str | None = None


# ---------------------------------------------------------------------------
# SovereignRuntime
# ---------------------------------------------------------------------------

class SovereignRuntime:
    """
    Central bootstrap that connects the four SPIO layers.

    Layers:
        PIO     – user-facing operator (sessions, middleware, intent)
        GPIA    – memory + reasoning engine
        ASIOS   – governor, government, pass broker, failsafe
        Moltbot – multi-channel gateway
    """

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()

        # Layers (populated by boot)
        self.wavelength_gate = None
        self.memory = None
        self.reasoning_engine = None
        self.asios = None
        self.pio = None
        self.gateway = None

        self._booted = False

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    def boot(self) -> None:
        """Instantiate and wire all four layers."""
        if self._booted:
            return

        # --- 1. Wavelength gate ---
        from core.wavelengths import WavelengthGate
        self.wavelength_gate = WavelengthGate(
            threshold=self.config.wavelength_threshold,
        )

        # --- 2. GPIA: Memory ---
        from pathlib import Path

        from gpia.memory import Memory, SimpleEmbedder

        persist = (
            Path(self.config.memory_persist_path)
            if self.config.memory_persist_path
            else None
        )

        # Use semantic embedder when an LLM provider is configured
        embedder = SimpleEmbedder()
        if self.config.llm_provider == "ollama":
            try:
                from gpia.memory import OllamaEmbedder
                embedder = OllamaEmbedder(
                    model=self.config.embedding_model,
                    host=self.config.llm_host,
                )
                # Probe connection — fall back to hash-based if Ollama is down
                embedder.embed("probe")
                logger.info("GPIA Memory: OllamaEmbedder active")
            except Exception as exc:
                logger.warning(
                    "OllamaEmbedder unavailable (%s), falling back to SimpleEmbedder",
                    exc,
                )
                embedder = SimpleEmbedder()
        elif self.config.llm_provider == "openai" and self.config.openai_api_key:
            try:
                from gpia.memory import OpenAIEmbedder
                embedder = OpenAIEmbedder(api_key=self.config.openai_api_key)
                logger.info("GPIA Memory: OpenAIEmbedder active")
            except Exception as exc:
                logger.warning(
                    "OpenAIEmbedder unavailable (%s), falling back to SimpleEmbedder",
                    exc,
                )
                embedder = SimpleEmbedder()

        self.memory = Memory(embedder=embedder, persist_path=persist)

        # --- 2b. GPIA: Reasoning (optional) ---
        if self.config.llm_provider != "echo":
            from gpia.reasoning import ModelConfig, ReasoningEngine
            if self.config.llm_provider == "ollama":
                model_cfg = ModelConfig.ollama(
                    model=self.config.llm_model,
                    host=self.config.llm_host,
                )
            elif self.config.llm_provider == "openai":
                model_cfg = ModelConfig.openai(
                    model=self.config.llm_model,
                    api_key=self.config.openai_api_key,
                )
            else:
                raise ValueError(
                    f"Unknown llm_provider: {self.config.llm_provider}"
                )
            self.reasoning_engine = ReasoningEngine(model_cfg)

        # --- 3. ASIOS ---
        from asios import ASIOSRuntime
        self.asios = ASIOSRuntime()

        # --- 4. PIO ---
        from pio.operator import (
            PIOOperator,
            logging_middleware,
            memory_store_middleware,
        )

        self.pio = PIOOperator()
        self.pio.set_memory(self.memory)
        if self.reasoning_engine is not None:
            self.pio.set_reasoning_engine(self.reasoning_engine)

        # Middleware: safety -> wavelength -> logging -> memory_store
        self.pio.use(self._asios_safety_middleware)
        self.pio.use(self._wavelength_audit_middleware)
        self.pio.use(logging_middleware)
        self.pio.use(memory_store_middleware)

        # --- 5. Moltbot Gateway + channel ---
        from moltbot.gateway import Gateway
        self.gateway = Gateway()

        channel = self._create_channel()
        self.gateway.register(channel)

        # Wire gateway -> PIO
        self.gateway.on_message(self._gateway_handler)

        # --- 6. PassBroker providers ---
        from asios.pass_protocol import NeedType
        self.asios.pass_broker.register_provider(
            NeedType.KNOWLEDGE, self._knowledge_provider,
        )
        self.asios.pass_broker.register_provider(
            NeedType.CAPABILITY, self._capability_provider,
        )

        self._booted = True
        logger.info("SovereignRuntime booted")

    # ------------------------------------------------------------------
    # Run / Shutdown
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the gateway (blocks until stopped)."""
        if not self._booted:
            self.boot()
        await self.gateway.start()

    async def shutdown(self) -> None:
        """Graceful teardown of all layers."""
        if self.gateway is not None:
            await self.gateway.stop()
        if self.reasoning_engine is not None:
            await self.reasoning_engine.close()
        logger.info("SovereignRuntime shut down")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Unified status from all four layers."""
        result: dict[str, Any] = {"booted": self._booted}
        if not self._booted:
            return result

        result["pio"] = self.pio.stats()
        result["gpia"] = {
            "memory": self.memory.stats(),
            "reasoning": self.reasoning_engine is not None,
        }
        result["asios"] = self.asios.get_status()
        result["moltbot"] = self.gateway.status()
        result["wavelength_gate"] = self.wavelength_gate.get_stats()
        return result

    # ------------------------------------------------------------------
    # Internal: channel factory
    # ------------------------------------------------------------------

    def _create_channel(self):
        """Create the configured channel."""
        if self.config.channel == "terminal":
            from moltbot.channels.terminal import TerminalChannel
            return TerminalChannel()
        raise ValueError(f"Unsupported channel: {self.config.channel}")

    # ------------------------------------------------------------------
    # Internal: gateway handler
    # ------------------------------------------------------------------

    async def _gateway_handler(self, message) -> str | None:
        """Route a Moltbot Message through PIO."""
        session_id = message.metadata.get(
            "session_id", f"moltbot-{message.sender}",
        )
        return await self.pio.process(
            session_id=session_id,
            user_input=message.content,
            channel=message.channel.value if hasattr(message.channel, "value") else str(message.channel),
            user_id=message.sender,
        )

    # ------------------------------------------------------------------
    # Internal: middleware
    # ------------------------------------------------------------------

    async def _asios_safety_middleware(self, pio, session, text):
        """Middleware: check ASIOS governor health before processing."""
        health = self.asios.governor.check_health()
        session.context["asios_health"] = health

        if health.get("critical"):
            errors = "; ".join(health.get("errors", []))
            logger.error("ASIOS critical stop: %s", errors)
            return f"[ASIOS] Critical stop — {errors}. Request blocked for hardware safety."

        if health.get("throttled"):
            warnings = "; ".join(health.get("warnings", []))
            logger.warning("ASIOS throttled: %s", warnings)
            # Throttled = warn but allow (degraded mode)
            session.context["asios_throttled"] = True
            session.context["asios_warnings"] = warnings

        return None  # continue pipeline

    async def _wavelength_audit_middleware(self, pio, session, text):
        """Middleware: run input through WavelengthGate for audit + enforcement."""
        gate_result = self.wavelength_gate.evaluate(text)
        session.context["wavelength"] = {
            "density": gate_result.density,
            "resonance": gate_result.resonance,
            "safe": gate_result.safe,
            "converged": gate_result.converged,
            "reason": gate_result.reason,
        }

        # Enforce safety: block if unsafe AND did not self-correct.
        # Only enforce when a real LLM provider is active — in echo mode the
        # hash-based embedder produces arbitrary densities, so enforcement
        # would be noise.
        enforce = (
            self.config.wavelength_enforce
            and self.config.llm_provider != "echo"
        )
        if enforce and not gate_result.safe and not gate_result.converged:
            logger.warning(
                "Wavelength gate blocked request: %s", gate_result.reason,
            )
            return (
                f"[WAVELENGTH] Request blocked — {gate_result.reason}. "
                "The cognitive pipeline could not converge to a safe state."
            )

        return None  # safe or self-corrected — continue pipeline

    # ------------------------------------------------------------------
    # Internal: PassBroker providers
    # ------------------------------------------------------------------

    def _knowledge_provider(self, capsule, need):
        """Provide knowledge from Memory."""
        results = self.memory.search(need.description, top_k=3)
        if results:
            return [
                {"content": entry.content, "similarity": sim}
                for entry, sim in results
            ]
        return None

    def _capability_provider(self, capsule, need):
        """Provide capability info from Government."""
        decision = self.asios.government.route(need.description)
        return {
            "minister": decision.minister.title,
            "confidence": decision.confidence,
        }
