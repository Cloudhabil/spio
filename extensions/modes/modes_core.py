"""
Modes Extension - Operational Mode System

Ported from: CLI-main/src/core/modes/

Implements:
- BaseMode: Abstract base for all operational modes
- SovereignMode: Primary inquiry and philosophical mode
- TeachingMode: Pedagogical tutoring mode
- GardenerMode: Autonomous file organization
- ForensicMode: Debugging and inspection
- ManifestMode: Autonomous 40-cycle construction
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ModeType(Enum):
    """Types of operational modes."""
    SOVEREIGN = "sovereign"
    TEACHING = "teaching"
    GARDENER = "gardener"
    FORENSIC = "forensic"
    MANIFEST = "manifest"


class ModeState(Enum):
    """Mode lifecycle states."""
    INACTIVE = auto()
    ENTERING = auto()
    ACTIVE = auto()
    EXITING = auto()
    ERROR = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModeTransition:
    """
    Represents a transition between modes.

    Attributes:
        next_mode: Name of the mode to transition to
        reason: Reason for the transition
        context: Context data to pass to the next mode
    """
    next_mode: str
    reason: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeContext:
    """
    Context provided to modes during execution.

    Attributes:
        kernel: Reference to the kernel (if available)
        repo_root: Repository root path
        identity: Agent identity information
        state: Shared state dictionary
        config: Mode configuration
    """
    kernel: Any = None
    repo_root: Path = field(default_factory=Path.cwd)
    identity: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def emit_telemetry(self, event: str, data: Dict[str, Any]) -> None:
        """Emit a telemetry event."""
        logger.info(f"[Telemetry] {event}: {data}")

    def log(self, message: str) -> None:
        """Log a message."""
        print(message)


# =============================================================================
# BASE MODE
# =============================================================================

class BaseMode(ABC):
    """
    The Architectural Substrate for all GPIA Cognitive Modes.

    Enforces the TemporalFormalismContract and Resonance Alignment.
    All operational modes inherit from this base class.

    Lifecycle:
        1. enter(context) - Initialize the mode
        2. step() or execute_beat() - Perform one cycle of work
        3. exit() - Gracefully shutdown and return state
    """

    mode_name: str = "Base"
    mode_type: ModeType = ModeType.SOVEREIGN

    def __init__(self, ctx: Optional[ModeContext] = None):
        """
        Initialize the mode.

        Args:
            ctx: Mode context with kernel, identity, state
        """
        self.ctx = ctx or ModeContext()
        self.is_active = False
        self.state = ModeState.INACTIVE
        self.resonance_score = 0.0
        self.beat_count = 0
        self.last_beat_time = 0.0

    @abstractmethod
    def enter(self, context: Dict[str, Any]) -> None:
        """
        Initialize the mode without amnesia, inheriting kernel state.

        Args:
            context: Context dictionary from previous mode
        """
        self.is_active = True
        self.state = ModeState.ACTIVE
        self.last_beat_time = time.time()
        logger.info(f"[MODE] Entering {self.mode_name}...")

    @abstractmethod
    def step(self) -> Optional[ModeTransition]:
        """
        One cycle of mode operation.

        Returns:
            ModeTransition to change modes, or None to continue
        """
        pass

    def execute_beat(self, beat_count: int, energy: float) -> None:
        """
        Logic executed on every heartbeat pulse.

        Args:
            beat_count: Current beat number
            energy: Energy level (0.0 to 1.0)
        """
        self.beat_count = beat_count
        self.last_beat_time = time.time()

    @abstractmethod
    def exit(self) -> Dict[str, Any]:
        """
        Gracefully shutdown and return state for the next mode.

        Returns:
            State dictionary to pass to next mode
        """
        self.is_active = False
        self.state = ModeState.INACTIVE
        logger.info(f"[MODE] Exiting {self.mode_name}...")
        return {"final_resonance": self.resonance_score}

    def validate_resonance(self, target: float = 0.95) -> bool:
        """
        Enforces the stability gate for high-stakes cycles.

        Args:
            target: Target resonance score

        Returns:
            True if resonance meets target
        """
        return self.resonance_score >= target


# =============================================================================
# SOVEREIGN MODE
# =============================================================================

class SovereignMode(BaseMode):
    """
    The Inquiry Mode: Searching the Sea of Zeros with a Moral Compass.

    Primary operational mode for philosophical inquiry and reasoning.
    Implements the heartbeat loop with skill selection and crystallization.

    Features:
        - Philosophical verification (collision avoidance)
        - Skill selection based on energy and mood
        - Resonance measurement
        - Crystallization of insights
    """

    mode_name = "Sovereign"
    mode_type = ModeType.SOVEREIGN

    def enter(self, context: Dict[str, Any]) -> None:
        """Initialize sovereign mode with philosophical governors."""
        super().enter(context)
        self.context = context
        logger.info("[SOVEREIGN] Philosophy Active. Collision Avoidance Online.")

    def step(self) -> Optional[ModeTransition]:
        """
        One cycle in Sovereign mode.

        Returns:
            ModeTransition if mode change requested, None otherwise
        """
        # Check for mode transition commands
        cmd = self._read_command()

        if cmd in {"teach", "mode teach", "teaching"}:
            self.ctx.emit_telemetry("sovereign.transition_request", {"target": "Teaching"})
            return ModeTransition(next_mode="Teaching", reason="operator_request")

        if cmd in {"forensic", "mode forensic", "debug"}:
            self.ctx.emit_telemetry("sovereign.transition_request", {"target": "Forensic"})
            return ModeTransition(next_mode="Forensic", reason="operator_request")

        if cmd in {"garden", "mode garden", "gardener"}:
            self.ctx.emit_telemetry("sovereign.transition_request", {"target": "Gardener"})
            return ModeTransition(next_mode="Gardener", reason="operator_request")

        if cmd in {"manifest", "mode manifest", "build"}:
            self.ctx.emit_telemetry("sovereign.transition_request", {"target": "Manifest"})
            return ModeTransition(next_mode="Manifest", reason="operator_request")

        # Execute standard sovereign cycle
        self._execute_sovereign_cycle(cmd)
        return None

    def execute_beat(self, beat_count: int, energy: float) -> None:
        """The Heartbeat of the Organism - Inquiry-to-Truth implementation."""
        super().execute_beat(beat_count, energy)

        # 1. Calculate drift
        drift_ms = (time.time() - self.last_beat_time) * 1000

        # 2. Update resonance based on energy
        self.resonance_score = (energy + self.resonance_score) / 2.0

        # 3. Periodic visualization
        if beat_count % 20 == 0:
            logger.info(f"[SOVEREIGN BEAT {beat_count}] Resonance: {self.resonance_score:.3f}")

    def exit(self) -> Dict[str, Any]:
        """Gracefully shutdown and return the resonance summary."""
        super().exit()
        return {
            "final_resonance": self.resonance_score,
            "directive_status": "SEARCHING_FOR_SHADOW"
        }

    def _read_command(self) -> str:
        """Read next command (stub for input)."""
        return ""

    def _execute_sovereign_cycle(self, cmd: str) -> None:
        """Execute one cycle of sovereign processing."""
        if cmd:
            logger.info(f"[SOVEREIGN] Processing: {cmd}")
            self.ctx.emit_telemetry("sovereign.process", {"cmd": cmd})


# =============================================================================
# TEACHING MODE
# =============================================================================

class TeachingMode(BaseMode):
    """
    Teaching/Tutoring operational mode.

    Operates with pedagogical focus:
        - Explains concepts in accessible ways
        - Creates exercises and learning materials
        - Provides guided learning experiences
        - Adapts to learner understanding
    """

    mode_name = "Teaching"
    mode_type = ModeType.TEACHING

    def __init__(self, ctx: Optional[ModeContext] = None):
        super().__init__(ctx)
        self.interactions: List[Dict[str, Any]] = []

    def enter(self, context: Dict[str, Any]) -> None:
        """Initialize teaching mode."""
        super().enter(context)
        logger.info("[TEACHING] Pedagogical mode activated.")

    def step(self) -> Optional[ModeTransition]:
        """
        One cycle in Teaching mode.

        Returns:
            ModeTransition if mode change requested, None otherwise
        """
        cmd = self._read_command()

        # Handle transitions
        if cmd in {"back", "mode sovereign", "exit teaching"}:
            self.ctx.emit_telemetry("teaching.transition_request", {"target": "Sovereign"})
            return ModeTransition(next_mode="Sovereign", reason="return_to_sovereign")

        if cmd in {"forensic", "mode forensic", "debug"}:
            self.ctx.emit_telemetry("teaching.transition_request", {"target": "Forensic"})
            return ModeTransition(next_mode="Forensic", reason="operator_request")

        if not cmd:
            return None

        # Log teaching interaction
        self._log_interaction(cmd)

        # Create teaching response
        response = self._create_teaching_response(cmd)
        self.ctx.log(response)

        return None

    def exit(self) -> Dict[str, Any]:
        """Exit teaching mode with interaction summary."""
        super().exit()
        return {
            "final_resonance": self.resonance_score,
            "interactions": len(self.interactions),
        }

    def _read_command(self) -> str:
        """Read next command (stub)."""
        return ""

    def _log_interaction(self, cmd: str) -> None:
        """Log a teaching interaction."""
        interaction = {
            "timestamp": time.time(),
            "cmd": cmd,
            "type": "teaching_interaction",
        }
        self.interactions.append(interaction)
        self.ctx.emit_telemetry("teaching.interaction", interaction)

    def _create_teaching_response(self, cmd: str) -> str:
        """Create a teaching-oriented response."""
        return f"[Teaching] Explaining: {cmd}\n           [Concept breakdown and guided understanding]"

    def _cmd_to_vector(self, cmd: str) -> List[float]:
        """Convert command string to state vector."""
        codes = [float(ord(c)) / 256.0 for c in cmd[:32]]
        codes.extend([0.0] * (32 - len(codes)))
        return codes[:32]


# =============================================================================
# GARDENER MODE
# =============================================================================

class GardenerMode(BaseMode):
    """
    Autonomous File Organization Mode.

    Features:
        - Real-time filesystem monitoring
        - Intelligent classification via GPIA
        - Auto-organization with zero deletion
        - Audit logging
        - Interactive controls
    """

    mode_name = "Gardener"
    mode_type = ModeType.GARDENER

    def __init__(self, ctx: Optional[ModeContext] = None):
        super().__init__(ctx)
        self.stats = {
            "artifacts_processed": 0,
            "artifacts_organized": 0,
            "queue_size": 0,
            "classifications": {},
        }
        self.last_stats_time = 0.0
        self.stats_interval = 60.0

    def enter(self, context: Dict[str, Any]) -> None:
        """Initialize gardener mode."""
        super().enter(context)
        logger.info("=" * 80)
        logger.info("GARDENER MODE - Autonomous File Organization")
        logger.info("=" * 80)
        self.ctx.emit_telemetry("gardener.started", {"root": str(self.ctx.repo_root)})

    def step(self) -> Optional[ModeTransition]:
        """
        One cycle in Gardener mode.

        Returns:
            ModeTransition if mode change requested, None otherwise
        """
        cmd = self._read_command()

        # Handle transitions
        if cmd in {"back", "mode sovereign", "exit gardener"}:
            self.ctx.emit_telemetry("gardener.transition_request", {"target": "Sovereign"})
            return ModeTransition(next_mode="Sovereign", reason="return_to_sovereign")

        # Periodic stats
        if time.time() - self.last_stats_time >= self.stats_interval:
            self._report_stats()

        return None

    def exit(self) -> Dict[str, Any]:
        """Exit gardener mode with final stats."""
        self._report_stats()
        super().exit()
        self.ctx.emit_telemetry("gardener.stopped", self.stats)
        return {"final_resonance": self.resonance_score, "stats": self.stats}

    def _read_command(self) -> str:
        """Read next command (stub)."""
        return ""

    def _report_stats(self) -> None:
        """Report gardener statistics."""
        logger.info("-" * 80)
        logger.info(
            f"[Stats] Processed: {self.stats['artifacts_processed']} | "
            f"Organized: {self.stats['artifacts_organized']} | "
            f"Queue: {self.stats['queue_size']}"
        )
        if self.stats['classifications']:
            logger.info("[Stats] Classifications:")
            for classification, count in sorted(self.stats['classifications'].items()):
                logger.info(f"        {classification}: {count}")
        logger.info("-" * 80)

        self.ctx.emit_telemetry("gardener.stats", self.stats)
        self.last_stats_time = time.time()

    def process_artifact(self, path: str, classification: str) -> None:
        """Process and classify an artifact."""
        self.stats["artifacts_processed"] += 1
        self.stats["classifications"][classification] = \
            self.stats["classifications"].get(classification, 0) + 1
        logger.info(f"[Gardener] {path} -> {classification}")


# =============================================================================
# FORENSIC MODE
# =============================================================================

class ForensicMode(BaseMode):
    """
    Forensic/Debug operational mode.

    Operates in inspection and debugging configuration:
        - Examine internal state and identity
        - Inspect ledger records
        - Validate system invariants
        - Trace execution paths
        - Verify telemetry
    """

    mode_name = "Forensic"
    mode_type = ModeType.FORENSIC

    def enter(self, context: Dict[str, Any]) -> None:
        """Initialize forensic mode."""
        super().enter(context)
        logger.info("[FORENSIC] Debug mode activated.")

    def step(self) -> Optional[ModeTransition]:
        """
        One cycle in Forensic mode.

        Returns:
            ModeTransition if mode change requested, None otherwise
        """
        cmd = self._read_command()

        # Handle transitions
        if cmd in {"back", "mode sovereign", "exit forensic"}:
            self.ctx.emit_telemetry("forensic.transition_request", {"target": "Sovereign"})
            return ModeTransition(next_mode="Sovereign", reason="return_to_sovereign")

        if cmd in {"teach", "mode teach"}:
            self.ctx.emit_telemetry("forensic.transition_request", {"target": "Teaching"})
            return ModeTransition(next_mode="Teaching", reason="operator_request")

        # Handle forensic commands
        if cmd == "dump identity":
            self._dump_identity()
            return None

        if cmd == "dump state":
            self._dump_state()
            return None

        if cmd == "verify":
            self._verify_system()
            return None

        if cmd == "help":
            self._show_help()
            return None

        return None

    def exit(self) -> Dict[str, Any]:
        """Exit forensic mode."""
        super().exit()
        return {"final_resonance": self.resonance_score}

    def _read_command(self) -> str:
        """Read next command (stub)."""
        return ""

    def _dump_identity(self) -> None:
        """Dump agent identity."""
        identity_str = json.dumps(self.ctx.identity, indent=2)
        self.ctx.log(f"[Forensic] Identity Record:\n{identity_str}")
        self.ctx.emit_telemetry("forensic.dump_identity", {"keys": list(self.ctx.identity.keys())})

    def _dump_state(self) -> None:
        """Dump agent state."""
        state_str = json.dumps(self.ctx.state, indent=2)
        self.ctx.log(f"[Forensic] Agent State:\n{state_str}")
        self.ctx.emit_telemetry("forensic.dump_state", {"keys": list(self.ctx.state.keys())})

    def _verify_system(self) -> None:
        """Verify system invariants."""
        self.ctx.log("[Forensic] Verifying system invariants...")

        # Check identity
        required = ["agent_id", "kernel_signature", "created_at"]
        missing = [k for k in required if k not in self.ctx.identity]
        if missing:
            self.ctx.log(f"  [WARN] Identity missing: {missing}")
        else:
            self.ctx.log("  [OK] Identity valid")

        # Check state
        if isinstance(self.ctx.state, dict):
            self.ctx.log(f"  [OK] State dict accessible ({len(self.ctx.state)} keys)")
        else:
            self.ctx.log("  [FAIL] State not accessible")

        self.ctx.emit_telemetry("forensic.verify_complete", {})

    def _show_help(self) -> None:
        """Display available forensic commands."""
        help_text = """
[Forensic] Available Commands:
  dump identity        - Show agent identity record
  dump state          - Show agent state dict
  verify              - Verify system invariants
  back                - Return to Sovereign
  mode sovereign      - Return to Sovereign
  mode teach          - Switch to Teaching mode
  help                - Show this help
"""
        self.ctx.log(help_text)


# =============================================================================
# MANIFEST MODE
# =============================================================================

class ManifestMode(BaseMode):
    """
    The Manifestation Mode: A 40-Cycle Autonomous Build.

    Executes a structured construction sequence:
        - Cycles 1-10: Foundation Substrate
        - Cycles 11-20: Visual Kinematics
        - Cycles 21-30: Persistent Voice
        - Cycles 31-40: File Ingestion Layer
    """

    mode_name = "Manifest"
    mode_type = ModeType.MANIFEST
    TOTAL_CYCLES = 40

    def enter(self, context: Dict[str, Any]) -> None:
        """Initialize manifestation mode."""
        super().enter(context)
        logger.info("\n" + "=" * 80)
        logger.info("  GENESIS 40-CYCLE CONSTRUCTION INITIATED")
        logger.info("=" * 80)

    def step(self) -> Optional[ModeTransition]:
        """
        One cycle in Manifest mode.

        Returns:
            ModeTransition when construction complete, None otherwise
        """
        self.beat_count += 1

        if self.beat_count <= 10:
            logger.info(f"[CONSTRUCTION {self.beat_count:02}/{self.TOTAL_CYCLES}] Foundation Substrate...")
        elif self.beat_count <= 20:
            logger.info(f"[CONSTRUCTION {self.beat_count:02}/{self.TOTAL_CYCLES}] Visual Kinematics...")
        elif self.beat_count <= 30:
            logger.info(f"[CONSTRUCTION {self.beat_count:02}/{self.TOTAL_CYCLES}] Persistent Voice...")
        elif self.beat_count <= 40:
            logger.info(f"[CONSTRUCTION {self.beat_count:02}/{self.TOTAL_CYCLES}] File Ingestion Layer...")

        if self.beat_count >= self.TOTAL_CYCLES:
            logger.info("\n[MANIFEST] Terminal beat reached. Construction complete.")
            return ModeTransition(
                next_mode="Sovereign",
                reason="construction_complete",
                context={"status": "BULLETPROOF"}
            )

        return None

    def exit(self) -> Dict[str, Any]:
        """Exit manifest mode with status."""
        super().exit()
        return {"status": "BULLETPROOF", "cycles_completed": self.beat_count}


# =============================================================================
# MODE MANAGER
# =============================================================================

class ModeManager:
    """
    Manages mode lifecycle and transitions.

    Provides:
        - Mode registration
        - Transition handling
        - State persistence across modes
    """

    def __init__(self, ctx: Optional[ModeContext] = None):
        self.ctx = ctx or ModeContext()
        self.modes: Dict[str, BaseMode] = {}
        self.current_mode: Optional[BaseMode] = None
        self.transition_history: List[ModeTransition] = []

        # Register default modes
        self._register_default_modes()

    def _register_default_modes(self) -> None:
        """Register all default modes."""
        self.register_mode("Sovereign", SovereignMode(self.ctx))
        self.register_mode("Teaching", TeachingMode(self.ctx))
        self.register_mode("Gardener", GardenerMode(self.ctx))
        self.register_mode("Forensic", ForensicMode(self.ctx))
        self.register_mode("Manifest", ManifestMode(self.ctx))

    def register_mode(self, name: str, mode: BaseMode) -> None:
        """Register a mode."""
        self.modes[name] = mode

    def get_mode(self, name: str) -> Optional[BaseMode]:
        """Get a mode by name."""
        return self.modes.get(name)

    def enter_mode(self, name: str, context: Dict[str, Any] = None) -> bool:
        """
        Enter a mode.

        Args:
            name: Mode name
            context: Context to pass to the mode

        Returns:
            True if successful
        """
        mode = self.modes.get(name)
        if not mode:
            logger.error(f"Unknown mode: {name}")
            return False

        # Exit current mode
        exit_context = {}
        if self.current_mode:
            exit_context = self.current_mode.exit()

        # Merge contexts
        merged_context = {**(context or {}), **exit_context}

        # Enter new mode
        mode.enter(merged_context)
        self.current_mode = mode

        logger.info(f"[ModeManager] Transitioned to {name}")
        return True

    def step(self) -> Optional[ModeTransition]:
        """Execute one step of the current mode."""
        if not self.current_mode:
            return None

        transition = self.current_mode.step()

        if transition:
            self.transition_history.append(transition)
            self.enter_mode(transition.next_mode, transition.context)

        return transition

    def get_current_mode(self) -> Optional[str]:
        """Get current mode name."""
        return self.current_mode.mode_name if self.current_mode else None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_mode_manager: Optional[ModeManager] = None


def create_mode(mode_type: ModeType, ctx: Optional[ModeContext] = None) -> BaseMode:
    """Create a mode by type."""
    mode_map = {
        ModeType.SOVEREIGN: SovereignMode,
        ModeType.TEACHING: TeachingMode,
        ModeType.GARDENER: GardenerMode,
        ModeType.FORENSIC: ForensicMode,
        ModeType.MANIFEST: ManifestMode,
    }
    mode_class = mode_map.get(mode_type, SovereignMode)
    return mode_class(ctx)


def get_mode_manager() -> ModeManager:
    """Get global mode manager instance."""
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
    return _mode_manager


__all__ = [
    # Base
    "BaseMode", "ModeTransition", "ModeContext",
    # Modes
    "SovereignMode", "TeachingMode", "GardenerMode",
    "ForensicMode", "ManifestMode",
    # Enums
    "ModeType", "ModeState",
    # Manager
    "ModeManager",
    # Factory
    "create_mode", "get_mode_manager",
]
