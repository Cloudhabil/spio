"""
Reflexes - Fast Safety Mechanisms

System 1 responses executing in sub-50ms for immediate safety decisions.

Each reflex:
1. Receives: task, context, level, flags, runtime
2. Returns: action, payload, audit trail
3. Executes in priority order (lower = earlier)
"""

from __future__ import annotations

import re
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# REFLEX TYPES
# =============================================================================

class ReflexAction(Enum):
    """Actions a reflex can return."""
    PASS = "PASS"                    # Allow next reflex
    DENY = "DENY"                    # Block request
    MODIFY_CONTEXT = "MODIFY_CONTEXT"  # Inject into context
    REPLY = "REPLY"                  # Return response
    STORE_MEMORY = "STORE_MEMORY"    # Save to memory
    SEARCH_MEMORY = "SEARCH_MEMORY"  # Query memory


class ReflexLayer(Enum):
    """Reflex execution layers."""
    L1 = "L1"  # Core reflexes
    L2 = "L2"  # Optimization
    L3 = "L3"  # Enhancement
    L4 = "L4"  # Governance


@dataclass
class ReflexResult:
    """Result from reflex execution."""
    action: ReflexAction
    payload: Dict[str, Any] = field(default_factory=dict)
    context_delta: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_result(cls, level: str = "L1") -> "ReflexResult":
        """Create a PASS result."""
        return cls(
            action=ReflexAction.PASS,
            audit={"level": level, "decision": "pass"}
        )

    @classmethod
    def deny_result(cls, reason: str, level: str = "L4") -> "ReflexResult":
        """Create a DENY result."""
        return cls(
            action=ReflexAction.DENY,
            payload={"reason": reason},
            audit={"level": level, "decision": "blocked"}
        )


@dataclass
class ReflexManifest:
    """Reflex configuration manifest."""
    id: str
    name: str
    layer: ReflexLayer
    priority: int
    enabled: bool = True
    version: str = "1.0.0"
    description: str = ""


# =============================================================================
# BASE REFLEX
# =============================================================================

class Reflex(ABC):
    """Base class for all reflexes."""

    def __init__(self, manifest: ReflexManifest):
        self.manifest = manifest

    @abstractmethod
    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Execute the reflex."""
        pass

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def priority(self) -> int:
        return self.manifest.priority


# =============================================================================
# REFLEX 1: ENGINE CORE (Priority 0, L1)
# =============================================================================

class EngineCore(Reflex):
    """
    Kernel reflex: no-op marker for boot sequencing.

    Always passes to allow subsequent reflexes to run.
    This is the first reflex executed in every chain.
    """

    def __init__(self):
        super().__init__(ReflexManifest(
            id="system/reflex-engine-core",
            name="Engine Core",
            layer=ReflexLayer.L1,
            priority=0,
            description="Kernel boot sequencing"
        ))

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Return PASS to allow subsequent reflexes."""
        return ReflexResult.pass_result(level)


# =============================================================================
# REFLEX 2: BIOMEDICAL PRECISION (Priority 5, L1)
# =============================================================================

class BiomedicalPrecision(Reflex):
    """
    Research accuracy reflex for biomedical content.

    Ensures precision in medical/scientific responses.
    """

    # Keywords requiring precision
    PRECISION_KEYWORDS = {
        "dosage", "medication", "diagnosis", "treatment",
        "symptom", "drug", "patient", "clinical",
        "surgery", "prescription", "contraindication"
    }

    def __init__(self):
        super().__init__(ReflexManifest(
            id="research/biomedical_precision",
            name="Biomedical Precision",
            layer=ReflexLayer.L1,
            priority=5,
            description="Research accuracy enforcement"
        ))

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Check for biomedical content requiring precision."""
        lowered = task.lower()

        # Check for precision keywords
        found_keywords = [k for k in self.PRECISION_KEYWORDS if k in lowered]

        if found_keywords:
            # Flag for precision mode
            return ReflexResult(
                action=ReflexAction.MODIFY_CONTEXT,
                context_delta={
                    "precision_mode": True,
                    "precision_keywords": found_keywords,
                },
                audit={
                    "level": level,
                    "decision": "precision_flagged",
                    "keywords": found_keywords,
                }
            )

        return ReflexResult.pass_result(level)


# =============================================================================
# REFLEX 3: RECENCY INJECTOR (Priority 10, L1)
# =============================================================================

class RecencyInjector(Reflex):
    """
    Memory injection reflex.

    Injects recent memories into context for continuity.
    """

    def __init__(self):
        super().__init__(ReflexManifest(
            id="memory/reflex-recency-injector",
            name="Recency Injector",
            layer=ReflexLayer.L1,
            priority=10,
            description="Inject recent memories"
        ))

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Inject recent memories into context."""
        lowered = (task or "").lower()

        # Check for memory commands
        if "memory" in lowered:
            if re.search(r"\b(save|store|remember)\b", lowered):
                fact = self._extract_fact(task)
                return ReflexResult(
                    action=ReflexAction.STORE_MEMORY,
                    payload={"fact": fact},
                    audit={"level": level, "decision": "store_memory"}
                )

            if "search memory" in lowered or "recall" in lowered:
                return ReflexResult(
                    action=ReflexAction.SEARCH_MEMORY,
                    payload={"content": task},
                    audit={"level": level, "decision": "search_memory"}
                )

        # Check if recency injection is allowed
        if not flags.get("allow_recency", True):
            return ReflexResult.pass_result(level)

        # Get recent memories from runtime
        memories = runtime.get("recent_memories", [])
        if not memories:
            return ReflexResult.pass_result(level)

        # Format and inject
        buffer = self._format_buffer(memories)
        return ReflexResult(
            action=ReflexAction.MODIFY_CONTEXT,
            context_delta={
                "reflex_recency": buffer,
                "reflex_recency_items": memories,
            },
            audit={"level": level, "decision": "inject_recency"}
        )

    def _extract_fact(self, task: str) -> str:
        """Extract quoted fact from task."""
        match = re.search(r"['\"]([^'\"]+)['\"]", task)
        if match:
            return f"{task}\nFact: {match.group(1)}"
        return task

    def _format_buffer(self, memories: List[Dict]) -> str:
        """Format memories as buffer."""
        lines = []
        for mem in memories:
            content = mem.get("content", "")
            if content:
                lines.append(f"- {content}")
        return "\n".join(lines)


# =============================================================================
# REFLEX 4: STABILIZER (Priority 20, L2)
# =============================================================================

class Stabilizer(Reflex):
    """
    Cache-based reuse of deterministic responses.

    Provides stability through response caching.
    """

    def __init__(self):
        super().__init__(ReflexManifest(
            id="optimization/reflex-stabilizer",
            name="Stabilizer",
            layer=ReflexLayer.L2,
            priority=20,
            description="Cache-based response reuse"
        ))
        self._cache: Dict[str, Dict] = {}

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Check cache and return if hit."""
        ttl = int(flags.get("stabilizer_ttl", 60))
        cache_key = self._hash(task, context)
        now = time.time()

        # Check cache
        entry = self._cache.get(cache_key)
        if entry and (now - entry["ts"]) <= ttl:
            return ReflexResult(
                action=ReflexAction.REPLY,
                payload={"response": entry["response"]},
                audit={"level": level, "decision": "cache_hit"}
            )

        # Seed cache if requested
        if flags.get("stabilizer_seed"):
            response = flags.get("stabilizer_response", "")
            if response:
                self._cache[cache_key] = {"response": response, "ts": now}
                return ReflexResult(
                    action=ReflexAction.PASS,
                    audit={"level": level, "decision": "cache_seeded"}
                )

        return ReflexResult.pass_result(level)

    def _hash(self, task: str, context: Dict) -> str:
        """Create cache key from task and context."""
        payload = json.dumps(
            {"task": task, "context": context},
            sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()


# =============================================================================
# REFLEX 5: GUARD L4 (Priority 90, L4)
# =============================================================================

class GuardL4(Reflex):
    """
    Deterministic safety and contract enforcement gate.

    Blocks requests matching deny patterns.
    """

    DEFAULT_DENY_PATTERNS = [
        "delete all",
        "drop database",
        "rm -rf",
        "format c:",
        "sudo rm",
        "shutdown",
        "destroy",
    ]

    def __init__(self, deny_patterns: Optional[List[str]] = None):
        super().__init__(ReflexManifest(
            id="governance/reflex-guard-l4",
            name="Guard L4",
            layer=ReflexLayer.L4,
            priority=90,
            description="Safety contract enforcement"
        ))
        self.deny_patterns = deny_patterns or self.DEFAULT_DENY_PATTERNS

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Validate task against deny patterns."""
        # Schema validation
        valid, reason = self._validate_schema(task, context, level, flags)
        if not valid:
            return ReflexResult.deny_result(reason, level)

        # Pattern matching
        lowered = task.lower()
        for pattern in self.deny_patterns:
            if pattern in lowered:
                return ReflexResult.deny_result("destructive_request", level)

        return ReflexResult.pass_result(level)

    def _validate_schema(
        self, task: str, context: Dict, level: str, flags: Dict
    ) -> tuple[bool, str]:
        """Validate input schema."""
        if not isinstance(task, str):
            return False, "task must be string"
        if not isinstance(context, dict):
            return False, "context must be object"
        if not isinstance(level, str):
            return False, "level must be string"
        if not isinstance(flags, dict):
            return False, "flags must be object"
        return True, ""


# =============================================================================
# REFLEX 6: AUDIT LOGGER (Priority 95, L4)
# =============================================================================

class AuditLogger(Reflex):
    """
    Audit trail logger for all reflex decisions.

    Records every decision for compliance and debugging.
    """

    def __init__(self):
        super().__init__(ReflexManifest(
            id="system/reflex-audit-logger",
            name="Audit Logger",
            layer=ReflexLayer.L4,
            priority=95,
            description="Audit trail logging"
        ))
        self.audit_log: List[Dict] = []

    def run(
        self,
        task: str,
        context: Dict[str, Any],
        level: str,
        flags: Dict[str, Any],
        runtime: Dict[str, Any],
    ) -> ReflexResult:
        """Log the audit trail."""
        # Get previous decisions from runtime
        decisions = runtime.get("reflex_decisions", [])

        entry = {
            "timestamp": time.time(),
            "task_hash": hashlib.sha256(task.encode()).hexdigest()[:16],
            "level": level,
            "decisions": decisions,
            "flags_count": len(flags),
        }

        self.audit_log.append(entry)

        # Keep last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

        return ReflexResult(
            action=ReflexAction.PASS,
            context_delta={"audit_entry": entry},
            audit={"level": level, "decision": "logged"}
        )

    def get_recent_audits(self, limit: int = 10) -> List[Dict]:
        """Get recent audit entries."""
        return self.audit_log[-limit:]


# =============================================================================
# REFLEX REGISTRY
# =============================================================================

class ReflexRegistry:
    """Registry of all available reflexes."""

    def __init__(self):
        self.reflexes: Dict[str, Reflex] = {}

    def register(self, reflex: Reflex):
        """Register a reflex."""
        self.reflexes[reflex.id] = reflex

    def get(self, reflex_id: str) -> Optional[Reflex]:
        """Get reflex by ID."""
        return self.reflexes.get(reflex_id)

    def get_all(self) -> List[Reflex]:
        """Get all reflexes sorted by priority."""
        return sorted(self.reflexes.values(), key=lambda r: r.priority)

    def get_by_layer(self, layer: ReflexLayer) -> List[Reflex]:
        """Get reflexes by layer."""
        return [r for r in self.get_all() if r.manifest.layer == layer]


# =============================================================================
# REFLEX RUNNER
# =============================================================================

class ReflexRunner:
    """
    Executes reflexes in priority order.

    Provides sub-50ms System 1 responses.
    """

    def __init__(self):
        self.registry = ReflexRegistry()
        self._register_defaults()

    def _register_defaults(self):
        """Register default reflexes."""
        self.registry.register(EngineCore())
        self.registry.register(BiomedicalPrecision())
        self.registry.register(RecencyInjector())
        self.registry.register(Stabilizer())
        self.registry.register(GuardL4())
        self.registry.register(AuditLogger())

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        level: str = "L1",
        flags: Optional[Dict[str, Any]] = None,
    ) -> ReflexResult:
        """
        Run all reflexes in priority order.

        Stops on first non-PASS action.
        """
        context = context or {}
        flags = flags or {}
        runtime = {"reflex_decisions": []}

        for reflex in self.registry.get_all():
            if not reflex.manifest.enabled:
                continue

            result = reflex.run(task, context, level, flags, runtime)

            # Track decision
            runtime["reflex_decisions"].append({
                "reflex": reflex.id,
                "action": result.action.value,
            })

            # Merge context deltas
            if result.context_delta:
                context.update(result.context_delta)

            # Stop on non-PASS
            if result.action != ReflexAction.PASS:
                return result

        # All passed
        return ReflexResult.pass_result(level)

    def get_stats(self) -> Dict:
        """Get runner statistics."""
        reflexes = self.registry.get_all()
        return {
            "total_reflexes": len(reflexes),
            "by_layer": {
                layer.value: len(self.registry.get_by_layer(layer))
                for layer in ReflexLayer
            },
            "reflexes": [
                {
                    "id": r.id,
                    "priority": r.priority,
                    "layer": r.manifest.layer.value,
                    "enabled": r.manifest.enabled,
                }
                for r in reflexes
            ],
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ReflexAction",
    "ReflexResult",
    "ReflexLayer",
    "ReflexManifest",
    "Reflex",
    "EngineCore",
    "BiomedicalPrecision",
    "RecencyInjector",
    "Stabilizer",
    "GuardL4",
    "AuditLogger",
    "ReflexRegistry",
    "ReflexRunner",
]
