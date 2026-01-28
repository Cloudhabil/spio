"""
PASS Protocol - Structured Agent-to-Agent Help Request System

PASS = Prevent, Assist, Synthesize, Solve

Enables agents to request help from other agents without prose explanations.
Implements a state machine for dependency resolution.

State Machine:
    ACTIVE → PASSED → ASSISTED → RESUMED → COMPLETED
                ↓
            FAILED (if no assistance available)
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pathlib import Path


class PassState(Enum):
    """States in the PASS protocol state machine."""
    ACTIVE = "active"           # Task is being worked on
    PASSED = "passed"           # Task passed to another agent
    ASSISTED = "assisted"       # Assistance received
    RESUMED = "resumed"         # Original agent resumed with help
    COMPLETED = "completed"     # Task completed
    FAILED = "failed"           # No assistance available


class NeedType(Enum):
    """Types of needs an agent can express."""
    KNOWLEDGE = "knowledge"     # Information or facts
    CAPABILITY = "capability"   # Skill or function
    FILE = "file"               # File access
    PERMISSION = "permission"   # Authorization
    RESOURCE = "resource"       # Computational resource
    CONTEXT = "context"         # Additional context


@dataclass
class Need:
    """A specific need expressed by an agent."""
    type: NeedType
    description: str
    priority: int = 1  # 1 = highest
    constraints: Dict[str, Any] = field(default_factory=dict)
    satisfied: bool = False
    satisfaction: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "description": self.description,
            "priority": self.priority,
            "constraints": self.constraints,
            "satisfied": self.satisfied,
        }


@dataclass
class Capsule:
    """
    A PASS Capsule - container for task state through the protocol.

    Capsules grow as assistance arrives, accumulating context
    and partial solutions.
    """
    id: str
    task: str
    state: PassState = PassState.ACTIVE
    needs: List[Need] = field(default_factory=list)
    assists: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None

    # Tracking
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    depth: int = 0  # Recursion depth
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Origin
    origin_agent: str = ""
    current_agent: str = ""

    def add_need(self, need: Need):
        """Add a need to the capsule."""
        self.needs.append(need)
        self.updated_at = time.time()

    def add_assist(self, agent_id: str, contribution: Any, need_index: int = -1):
        """Add assistance to the capsule."""
        assist = {
            "agent_id": agent_id,
            "contribution": contribution,
            "timestamp": time.time(),
            "need_index": need_index,
        }
        self.assists.append(assist)

        # Mark need as satisfied if specified
        if 0 <= need_index < len(self.needs):
            self.needs[need_index].satisfied = True
            self.needs[need_index].satisfaction = contribution

        self.updated_at = time.time()

    def unsatisfied_needs(self) -> List[Need]:
        """Get list of unsatisfied needs."""
        return [n for n in self.needs if not n.satisfied]

    def all_needs_satisfied(self) -> bool:
        """Check if all needs are satisfied."""
        return all(n.satisfied for n in self.needs)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "state": self.state.value,
            "needs": [n.to_dict() for n in self.needs],
            "assists": self.assists,
            "context": self.context,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "origin_agent": self.origin_agent,
            "current_agent": self.current_agent,
        }


# Type for assistance providers
AssistanceProvider = Callable[[Capsule, Need], Optional[Any]]


class PassBroker:
    """
    PASS Protocol Broker.

    Manages capsule lifecycle and routes assistance requests
    between agents.

    Example:
        broker = PassBroker()

        # Register assistance providers
        broker.register_provider(NeedType.KNOWLEDGE, knowledge_agent)
        broker.register_provider(NeedType.CAPABILITY, capability_agent)

        # Create a capsule
        capsule = broker.create_capsule("Translate this text to French", "agent_1")

        # Express needs
        broker.express_need(capsule.id, Need(
            type=NeedType.CAPABILITY,
            description="French language translation"
        ))

        # Request assistance
        await broker.request_assistance(capsule.id)
    """

    MAX_DEPTH = 5  # Maximum recursion depth

    def __init__(self):
        self.capsules: Dict[str, Capsule] = {}
        self.providers: Dict[NeedType, List[AssistanceProvider]] = {
            t: [] for t in NeedType
        }

        # Statistics
        self.total_capsules = 0
        self.successful_resolutions = 0
        self.failed_resolutions = 0

    def register_provider(self, need_type: NeedType, provider: AssistanceProvider):
        """Register an assistance provider for a need type."""
        self.providers[need_type].append(provider)

    def create_capsule(
        self,
        task: str,
        agent_id: str,
        parent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Capsule:
        """Create a new PASS capsule."""
        capsule_id = str(uuid.uuid4())[:8]

        # Calculate depth
        depth = 0
        if parent_id and parent_id in self.capsules:
            depth = self.capsules[parent_id].depth + 1
            self.capsules[parent_id].children_ids.append(capsule_id)

        capsule = Capsule(
            id=capsule_id,
            task=task,
            origin_agent=agent_id,
            current_agent=agent_id,
            parent_id=parent_id,
            depth=depth,
            context=context or {},
        )

        self.capsules[capsule_id] = capsule
        self.total_capsules += 1

        return capsule

    def get_capsule(self, capsule_id: str) -> Optional[Capsule]:
        """Get a capsule by ID."""
        return self.capsules.get(capsule_id)

    def express_need(self, capsule_id: str, need: Need) -> bool:
        """Express a need on a capsule."""
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return False

        capsule.add_need(need)
        return True

    def pass_capsule(self, capsule_id: str, to_agent: str) -> bool:
        """Pass a capsule to another agent."""
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return False

        capsule.state = PassState.PASSED
        capsule.current_agent = to_agent
        capsule.updated_at = time.time()

        return True

    async def request_assistance(self, capsule_id: str) -> bool:
        """
        Request assistance for unsatisfied needs.

        Iterates through providers until needs are satisfied
        or all providers exhausted.
        """
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return False

        # Check depth limit
        if capsule.depth >= self.MAX_DEPTH:
            capsule.state = PassState.FAILED
            capsule.context["failure_reason"] = "Max recursion depth exceeded"
            self.failed_resolutions += 1
            return False

        # Mark as passed
        capsule.state = PassState.PASSED

        # Try to satisfy each need
        for i, need in enumerate(capsule.needs):
            if need.satisfied:
                continue

            providers = self.providers.get(need.type, [])
            for provider in providers:
                try:
                    result = provider(capsule, need)
                    if result is not None:
                        capsule.add_assist(
                            agent_id=f"provider_{need.type.value}",
                            contribution=result,
                            need_index=i,
                        )
                        break
                except Exception as e:
                    capsule.context.setdefault("errors", []).append(str(e))

        # Check if all needs satisfied
        if capsule.all_needs_satisfied():
            capsule.state = PassState.ASSISTED
            return True
        else:
            capsule.state = PassState.FAILED
            capsule.context["failure_reason"] = "Not all needs could be satisfied"
            self.failed_resolutions += 1
            return False

    def resume_capsule(self, capsule_id: str) -> bool:
        """Resume a capsule after receiving assistance."""
        capsule = self.get_capsule(capsule_id)
        if not capsule or capsule.state != PassState.ASSISTED:
            return False

        capsule.state = PassState.RESUMED
        capsule.current_agent = capsule.origin_agent
        capsule.updated_at = time.time()

        return True

    def complete_capsule(self, capsule_id: str, result: Any) -> bool:
        """Mark a capsule as completed."""
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return False

        capsule.state = PassState.COMPLETED
        capsule.result = result
        capsule.updated_at = time.time()
        self.successful_resolutions += 1

        return True

    def fail_capsule(self, capsule_id: str, reason: str) -> bool:
        """Mark a capsule as failed."""
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return False

        capsule.state = PassState.FAILED
        capsule.context["failure_reason"] = reason
        capsule.updated_at = time.time()
        self.failed_resolutions += 1

        return True

    def get_chain(self, capsule_id: str) -> List[Capsule]:
        """Get the full chain of capsules from root to this one."""
        capsule = self.get_capsule(capsule_id)
        if not capsule:
            return []

        chain = [capsule]
        while capsule.parent_id:
            parent = self.get_capsule(capsule.parent_id)
            if not parent:
                break
            chain.insert(0, parent)
            capsule = parent

        return chain

    def stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        states = {}
        for capsule in self.capsules.values():
            states[capsule.state.value] = states.get(capsule.state.value, 0) + 1

        return {
            "total_capsules": self.total_capsules,
            "active_capsules": len(self.capsules),
            "successful_resolutions": self.successful_resolutions,
            "failed_resolutions": self.failed_resolutions,
            "resolution_rate": (
                self.successful_resolutions /
                max(1, self.successful_resolutions + self.failed_resolutions)
            ),
            "states": states,
            "providers": {t.value: len(p) for t, p in self.providers.items()},
        }
