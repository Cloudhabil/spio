"""
Reflex Engine - Pre-boot System 1

Deterministic reflex execution chain with sub-50ms response.
Handles immediate responses before full reasoning kicks in.

Features:
- YAML-based reflex registry
- Priority-based execution
- Policy and schema validation
- Audit logging
"""

import time
import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path


class ReflexPriority(Enum):
    """Reflex execution priority (lower = higher priority)."""
    CRITICAL = 0    # Safety reflexes
    HIGH = 1        # Security reflexes
    MEDIUM = 2      # Standard reflexes
    LOW = 3         # Convenience reflexes


class ReflexResult(Enum):
    """Result of reflex execution."""
    TRIGGERED = "triggered"     # Reflex handled the input
    PASSED = "passed"           # Reflex did not apply
    BLOCKED = "blocked"         # Reflex blocked the action
    ERROR = "error"             # Reflex encountered an error


@dataclass
class Reflex:
    """A single reflex definition."""
    id: str
    name: str
    description: str
    priority: ReflexPriority
    pattern: str  # Trigger pattern (regex or keyword)
    action: str   # Action to take
    response: Optional[str] = None  # Canned response if any

    # Constraints
    enabled: bool = True
    cooldown_ms: int = 0  # Minimum time between triggers
    max_triggers_per_minute: int = 0  # 0 = unlimited

    # Runtime state
    last_triggered: float = 0
    trigger_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.value,
            "pattern": self.pattern,
            "action": self.action,
            "enabled": self.enabled,
            "trigger_count": self.trigger_count,
        }


@dataclass
class ReflexAuditEntry:
    """Audit log entry for reflex execution."""
    timestamp: float
    reflex_id: str
    input_hash: str
    result: ReflexResult
    latency_ms: float
    response: Optional[str] = None


class ReflexEngine:
    """
    Reflex Engine - System 1 Processing.

    Executes deterministic reflexes before full reasoning.
    Designed for <50ms response time.

    Example:
        engine = ReflexEngine()
        engine.load_registry("reflexes/registry.yaml")

        # Process input
        result = engine.process("hello")
        if result.result == ReflexResult.TRIGGERED:
            print(result.response)  # "Hello! How can I help?"
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        max_latency_ms: float = 50.0,
    ):
        self.registry_path = registry_path
        self.max_latency_ms = max_latency_ms

        self.reflexes: Dict[str, Reflex] = {}
        self.priority_order: List[str] = []  # Sorted reflex IDs

        # Audit log
        self.audit_log: List[ReflexAuditEntry] = []
        self.max_audit_entries = 1000

        # Custom action handlers
        self._handlers: Dict[str, Callable] = {}

        # Statistics
        self.total_processed = 0
        self.total_triggered = 0
        self.total_blocked = 0
        self.avg_latency_ms = 0.0

        # Load default reflexes
        self._load_defaults()

        if registry_path and registry_path.exists():
            self.load_registry(registry_path)

    def _load_defaults(self):
        """Load default safety reflexes."""
        defaults = [
            Reflex(
                id="safety_stop",
                name="Emergency Stop",
                description="Immediate stop on safety keywords",
                priority=ReflexPriority.CRITICAL,
                pattern="(stop|halt|emergency|abort)",
                action="block",
                response="Action halted for safety.",
            ),
            Reflex(
                id="greeting",
                name="Greeting Response",
                description="Quick greeting response",
                priority=ReflexPriority.LOW,
                pattern="^(hello|hi|hey)$",
                action="respond",
                response="Hello! How can I help you today?",
            ),
            Reflex(
                id="thanks",
                name="Thanks Response",
                description="Quick thanks response",
                priority=ReflexPriority.LOW,
                pattern="^(thanks|thank you)$",
                action="respond",
                response="You're welcome!",
            ),
            Reflex(
                id="help",
                name="Help Response",
                description="Quick help response",
                priority=ReflexPriority.MEDIUM,
                pattern="^(help|\\?)$",
                action="respond",
                response="I'm Sovereign PIO. Ask me anything or say 'help' for more options.",
            ),
        ]

        for reflex in defaults:
            self.register(reflex)

    def register(self, reflex: Reflex):
        """Register a reflex."""
        self.reflexes[reflex.id] = reflex
        self._rebuild_priority_order()

    def unregister(self, reflex_id: str):
        """Unregister a reflex."""
        self.reflexes.pop(reflex_id, None)
        self._rebuild_priority_order()

    def _rebuild_priority_order(self):
        """Rebuild the priority-sorted execution order."""
        self.priority_order = sorted(
            self.reflexes.keys(),
            key=lambda x: (self.reflexes[x].priority.value, x)
        )

    def register_handler(self, action: str, handler: Callable):
        """Register a custom action handler."""
        self._handlers[action] = handler

    def load_registry(self, path: Path):
        """Load reflexes from YAML registry."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for reflex_data in data.get("reflexes", []):
            priority = ReflexPriority[reflex_data.get("priority", "MEDIUM").upper()]
            reflex = Reflex(
                id=reflex_data["id"],
                name=reflex_data.get("name", reflex_data["id"]),
                description=reflex_data.get("description", ""),
                priority=priority,
                pattern=reflex_data["pattern"],
                action=reflex_data["action"],
                response=reflex_data.get("response"),
                enabled=reflex_data.get("enabled", True),
                cooldown_ms=reflex_data.get("cooldown_ms", 0),
            )
            self.register(reflex)

    def _matches(self, reflex: Reflex, text: str) -> bool:
        """Check if input matches reflex pattern."""
        import re
        try:
            return bool(re.search(reflex.pattern, text, re.IGNORECASE))
        except re.error:
            # Fallback to simple contains
            return reflex.pattern.lower() in text.lower()

    def _check_cooldown(self, reflex: Reflex) -> bool:
        """Check if reflex is on cooldown."""
        if reflex.cooldown_ms <= 0:
            return False

        elapsed = (time.time() - reflex.last_triggered) * 1000
        return elapsed < reflex.cooldown_ms

    def _execute_action(
        self,
        reflex: Reflex,
        text: str,
    ) -> tuple[ReflexResult, Optional[str]]:
        """Execute the reflex action."""
        action = reflex.action

        if action == "block":
            return ReflexResult.BLOCKED, reflex.response

        if action == "respond":
            return ReflexResult.TRIGGERED, reflex.response

        if action == "pass":
            return ReflexResult.PASSED, None

        # Check for custom handler
        if action in self._handlers:
            try:
                response = self._handlers[action](text, reflex)
                return ReflexResult.TRIGGERED, response
            except Exception as e:
                return ReflexResult.ERROR, str(e)

        return ReflexResult.PASSED, None

    def process(self, text: str) -> tuple[ReflexResult, Optional[str], float]:
        """
        Process input through the reflex engine.

        Args:
            text: Input text to process

        Returns:
            Tuple of (result, response, latency_ms)
        """
        start_time = time.perf_counter()
        self.total_processed += 1

        result = ReflexResult.PASSED
        response = None
        triggered_reflex = None

        # Process in priority order
        for reflex_id in self.priority_order:
            reflex = self.reflexes[reflex_id]

            if not reflex.enabled:
                continue

            if self._check_cooldown(reflex):
                continue

            if not self._matches(reflex, text):
                continue

            # Execute action
            result, response = self._execute_action(reflex, text)

            if result in (ReflexResult.TRIGGERED, ReflexResult.BLOCKED):
                triggered_reflex = reflex
                reflex.last_triggered = time.time()
                reflex.trigger_count += 1

                if result == ReflexResult.TRIGGERED:
                    self.total_triggered += 1
                else:
                    self.total_blocked += 1

                break

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Update average latency
        n = self.total_processed
        self.avg_latency_ms = ((self.avg_latency_ms * (n - 1)) + latency_ms) / n

        # Audit log
        import hashlib
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:8]

        entry = ReflexAuditEntry(
            timestamp=time.time(),
            reflex_id=triggered_reflex.id if triggered_reflex else "",
            input_hash=input_hash,
            result=result,
            latency_ms=latency_ms,
            response=response,
        )
        self.audit_log.append(entry)

        # Trim audit log
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]

        return result, response, latency_ms

    def list_reflexes(self) -> List[Dict[str, Any]]:
        """List all registered reflexes."""
        return [
            self.reflexes[rid].to_dict()
            for rid in self.priority_order
        ]

    def stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_reflexes": len(self.reflexes),
            "enabled_reflexes": sum(1 for r in self.reflexes.values() if r.enabled),
            "total_processed": self.total_processed,
            "total_triggered": self.total_triggered,
            "total_blocked": self.total_blocked,
            "trigger_rate": self.total_triggered / max(1, self.total_processed),
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_target_ms": self.max_latency_ms,
        }
