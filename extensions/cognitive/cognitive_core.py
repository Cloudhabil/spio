"""
Cognitive Extension - Meta-Cortex and Cognitive Systems

Ported from: CLI-main/src/core/meta_cortex.py

Implements:
- MetaCortex: Introspection and self-model
- Thought packet observation
- Reflex improvement proposals
- Alignment tracking
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ThoughtPacket:
    """
    A single thought packet representing cognitive activity.

    Attributes:
        thought_id: Unique identifier for this thought
        origin_skill: Skill that generated this thought
        confidence_score: Self-reported confidence (0.0 to 1.0)
        content: The actual thought content
        timestamp: When the thought was generated
    """
    thought_id: str
    origin_skill: str
    confidence_score: float
    content: Any = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thought_id": self.thought_id,
            "origin_skill": self.origin_skill,
            "confidence_score": self.confidence_score,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class SelfModel:
    """
    The agent's model of itself - strengths, weaknesses, corrections.

    Attributes:
        version: Model version number
        strengths: List of observed strengths
        weaknesses: List of observed weaknesses
        active_patches: Currently active behavioral patches
        last_updated: Timestamp of last update
    """
    version: float = 1.0
    strengths: List[Dict[str, Any]] = field(default_factory=list)
    weaknesses: List[Dict[str, Any]] = field(default_factory=list)
    active_patches: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "self_perception": {
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
            },
            "correction_state": {
                "active_patches": self.active_patches,
                "last_updated": self.last_updated,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfModel':
        """Create from dictionary."""
        perception = data.get("self_perception", {})
        correction = data.get("correction_state", {})
        return cls(
            version=data.get("version", 1.0),
            strengths=perception.get("strengths", []),
            weaknesses=perception.get("weaknesses", []),
            active_patches=correction.get("active_patches", []),
            last_updated=correction.get("last_updated", time.time()),
        )


# =============================================================================
# ALIGNMENT TRACKER
# =============================================================================

class AlignmentTracker:
    """
    Tracks alignment conflicts between intent and outcome.

    When the agent's confidence (intent) differs significantly from
    actual outcomes, this may indicate an alignment issue.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.conflicts: List[Dict[str, Any]] = []

    def log_conflict(
        self,
        entry: Dict[str, Any],
        delta: float
    ) -> None:
        """Log an alignment conflict."""
        payload = {
            "ts": time.time(),
            "thought_id": entry.get("ts"),
            "delta": delta,
            "volition": entry.get("delta"),
            "outcome": "overestimated",
        }
        self.conflicts.append(payload)

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError as e:
            logger.error(f"Failed to log alignment conflict: {e}")

    def get_recent_conflicts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent alignment conflicts."""
        return self.conflicts[-limit:] if self.conflicts else []

    @property
    def conflict_rate(self) -> float:
        """Calculate conflict rate (conflicts per thought)."""
        return len(self.conflicts)


# =============================================================================
# REFLEX PROPOSER
# =============================================================================

class ReflexProposer:
    """
    Proposes behavioral patches when weaknesses are detected.

    A "reflex" is a fast, automatic response pattern. When the agent
    detects it's consistently wrong about something, it can propose
    a patch to adjust its reflexes.
    """

    def __init__(self):
        self.proposed_patches: List[Dict[str, Any]] = []
        self.validated_patches: List[Dict[str, Any]] = []

    def propose_patch(self, weakness: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Propose a patch for a detected weakness.

        Args:
            weakness: Dictionary describing the weakness

        Returns:
            Proposed patch or None if no patch can be generated
        """
        skill_id = weakness.get("skill", "unknown")
        delta = weakness.get("delta", 0.0)

        # Generate patch based on weakness type
        patch = {
            "patch_id": f"patch_{skill_id}_{int(time.time())}",
            "target_skill": skill_id,
            "correction_factor": 1.0 - delta * 0.5,  # Reduce confidence proportionally
            "created_at": time.time(),
            "applied": False,
            "weakness": weakness,
        }

        self.proposed_patches.append(patch)
        return patch

    def validate_patch(self, patch: Dict[str, Any]) -> bool:
        """
        Validate a proposed patch through shadow testing.

        In a full implementation, this would run the patch in a
        sandboxed environment to verify it doesn't break anything.

        Args:
            patch: The patch to validate

        Returns:
            True if patch is valid, False otherwise
        """
        # Simple validation - accept all patches with reasonable correction factors
        correction = patch.get("correction_factor", 0.0)
        is_valid = 0.1 <= correction <= 1.0

        if is_valid:
            self.validated_patches.append(patch)

        return is_valid

    def get_pending_patches(self) -> List[Dict[str, Any]]:
        """Get patches that haven't been applied yet."""
        return [p for p in self.proposed_patches if not p.get("applied")]


# =============================================================================
# META CORTEX
# =============================================================================

class MetaCortex:
    """
    Observes thought packets and proposes reflex improvements.

    The MetaCortex is the agent's "thinking about thinking" system.
    It monitors cognitive activity, detects patterns, and proposes
    improvements when it notices systematic errors.

    Components:
        - SelfModel: The agent's model of itself
        - AlignmentTracker: Monitors intent vs outcome alignment
        - ReflexProposer: Proposes behavioral patches
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None
    ):
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.state_dir = state_dir or (self.root_dir / "memory" / "meta_state_v1")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.self_model_path = self.state_dir / "self_model.json"
        self.alignment_log_path = self.state_dir / "alignment_tracker.log"
        self.thought_log_path = self.state_dir / "thought_packets.jsonl"

        # Components
        self.alignment_tracker = AlignmentTracker(self.alignment_log_path)
        self.reflex_proposer = ReflexProposer()

        # State
        self.self_model = self._load_self_model()
        self.thought_history: List[ThoughtPacket] = []

    def _load_self_model(self) -> SelfModel:
        """Load or create self model."""
        if self.self_model_path.exists():
            try:
                data = json.loads(self.self_model_path.read_text(encoding="utf-8"))
                return SelfModel.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load self model: {e}")

        # Create new self model
        model = SelfModel()
        self._save_self_model(model)
        return model

    def _save_self_model(self, model: SelfModel) -> None:
        """Save self model to disk."""
        try:
            self.self_model_path.parent.mkdir(parents=True, exist_ok=True)
            self.self_model_path.write_text(
                json.dumps(model.to_dict(), indent=2),
                encoding="utf-8"
            )
        except OSError as e:
            logger.error(f"Failed to save self model: {e}")

    def observe_thought(
        self,
        packet: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> float:
        """
        Compute delta between intent and outcome, update self model, and propose fixes.

        Args:
            packet: The thought packet (intent)
            outcome: The actual outcome

        Returns:
            The delta (difference) between intent and outcome
        """
        intent = float(packet.get("confidence_score", 0.5))
        reality = float(outcome.get("success_score", 0.5))
        delta = abs(intent - reality)

        # Log the thought
        self._log_thought(packet, outcome, delta)

        # If delta is significant, update self model and propose patch
        if delta > 0.3:
            weakness = self._update_self_model(
                packet.get("origin_skill", "unknown"),
                delta,
                reality < intent  # Did we overestimate?
            )
            if weakness:
                patch = self.reflex_proposer.propose_patch(weakness)
                if patch:
                    self.reflex_proposer.validate_patch(patch)

        return delta

    def _log_thought(
        self,
        packet: Dict[str, Any],
        outcome: Dict[str, Any],
        delta: float
    ) -> None:
        """Log thought packet to disk."""
        entry = {
            "ts": time.time(),
            "thought_id": packet.get("thought_id"),
            "origin_skill": packet.get("origin_skill"),
            "confidence_score": packet.get("confidence_score"),
            "success_score": outcome.get("success_score"),
            "delta": delta,
        }

        try:
            self.thought_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.thought_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except OSError as e:
            logger.error(f"Failed to log thought: {e}")

    def _update_self_model(
        self,
        skill_id: str,
        delta: float,
        overestimated: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Update self model based on observed discrepancy.

        Args:
            skill_id: The skill that generated the thought
            delta: The observed discrepancy
            overestimated: Whether we were overconfident

        Returns:
            Weakness entry if we overestimated, None otherwise
        """
        entry = {"skill": skill_id, "delta": delta, "ts": time.time()}

        if overestimated:
            self.self_model.weaknesses.append(entry)
            self.alignment_tracker.log_conflict(entry, delta)
        else:
            self.self_model.strengths.append(entry)

        self.self_model.last_updated = time.time()
        self._save_self_model(self.self_model)

        return entry if overestimated else None

    def get_status(self) -> Dict[str, Any]:
        """Get current metacognitive status."""
        return {
            "model_version": self.self_model.version,
            "strengths_count": len(self.self_model.strengths),
            "weaknesses_count": len(self.self_model.weaknesses),
            "active_patches": len(self.self_model.active_patches),
            "pending_patches": len(self.reflex_proposer.get_pending_patches()),
            "alignment_conflicts": self.alignment_tracker.conflict_rate,
            "last_updated": self.self_model.last_updated,
        }

    def apply_patch(self, patch_id: str) -> bool:
        """
        Apply a validated patch to the self model.

        Args:
            patch_id: ID of the patch to apply

        Returns:
            True if patch was applied, False otherwise
        """
        for patch in self.reflex_proposer.validated_patches:
            if patch.get("patch_id") == patch_id and not patch.get("applied"):
                patch["applied"] = True
                patch["applied_at"] = time.time()
                self.self_model.active_patches.append(patch)
                self._save_self_model(self.self_model)
                return True
        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_meta_cortex: Optional[MetaCortex] = None


def create_meta_cortex(
    root_dir: Optional[Path] = None,
    state_dir: Optional[Path] = None
) -> MetaCortex:
    """Create a new MetaCortex instance."""
    return MetaCortex(root_dir=root_dir, state_dir=state_dir)


def get_meta_cortex() -> MetaCortex:
    """Get global MetaCortex instance."""
    global _meta_cortex
    if _meta_cortex is None:
        _meta_cortex = MetaCortex()
    return _meta_cortex


__all__ = [
    # Core
    "MetaCortex", "ThoughtPacket", "SelfModel",
    # Analysis
    "AlignmentTracker", "ReflexProposer",
    # Factory
    "create_meta_cortex", "get_meta_cortex",
]
