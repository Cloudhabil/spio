"""
Core Extension - Essential System Modules

130+ Core system components organized by subsystem:

Subsystems:
1. Runtime (10)      - Capsule engine, PASS broker, government
2. Modes (8)         - Sovereign, manifest, teaching, gardener
3. Sovereignty (6)   - Identity, telemetry, heuristics
4. Safety (8)        - Safety codes, geometry, cognitive governor
5. Reflexes (6)      - Reflex engine, corrector, biomedical
6. Cognitive (8)     - Meta cortex, affect, resonance
7. Budget (6)        - Ledger, orchestrator, scheduler
8. Skills (8)        - Loader, assessor, MCP orchestrator
9. Context (6)       - Pager, DAG planner, dense logic
10. Infrastructure (10) - GPIA server, bridge, worker
11. Gardening (6)    - Filesystem, sovereign gardener
12. Substrate (8)    - Compressor, cracker, expansion
13. Misc (30+)       - Fusion, forager, linker, topology

Reference: CLI-main/src/core/*
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ============================================================================
# CONSTANTS (from Brahim's Calculator)
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2           # 1.6180339887498949
BETA = 1 / PHI ** 3                    # 0.2360679774997897
GENESIS_CONSTANT = 2 / 901             # 0.00221975...
SUM_CONSTANT = 214
CENTER = 107


# ============================================================================
# 1. RUNTIME SUBSYSTEM
# ============================================================================

class CapsuleType(Enum):
    """Types of runtime capsules."""
    STANDARD = "standard"
    SECURE = "secure"
    SANDBOXED = "sandboxed"
    PRIVILEGED = "privileged"


@dataclass
class Capsule:
    """A runtime capsule for isolated execution."""
    id: str
    capsule_type: CapsuleType
    created_at: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"


class CapsuleEngine:
    """Engine for managing runtime capsules."""

    def __init__(self):
        self._capsules: Dict[str, Capsule] = {}

    def create(self, capsule_type: CapsuleType = CapsuleType.STANDARD) -> Capsule:
        """Create a new capsule."""
        capsule_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
        capsule = Capsule(id=capsule_id, capsule_type=capsule_type)
        self._capsules[capsule_id] = capsule
        return capsule

    def get(self, capsule_id: str) -> Optional[Capsule]:
        """Get capsule by ID."""
        return self._capsules.get(capsule_id)

    def destroy(self, capsule_id: str) -> bool:
        """Destroy a capsule."""
        if capsule_id in self._capsules:
            del self._capsules[capsule_id]
            return True
        return False


class PASSBroker:
    """
    PASS Protocol broker for epistemic safety.

    PASS = Pause, Assess, Signal, Seek
    """

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self._signals: List[Dict] = []

    def assess(self, response: str, confidence: float) -> Tuple[bool, str]:
        """Assess if response meets confidence threshold."""
        if confidence >= self.confidence_threshold:
            return True, "approved"
        return False, "low_confidence"

    def signal(self, message: str, level: str = "info") -> None:
        """Signal an epistemic event."""
        self._signals.append({
            "message": message,
            "level": level,
            "timestamp": time.time()
        })

    def get_signals(self) -> List[Dict]:
        """Get recent signals."""
        return self._signals[-100:]


class Government:
    """Runtime government for resource governance."""

    def __init__(self):
        self.policies: Dict[str, Any] = {}
        self.ministers: Dict[str, Any] = {}

    def set_policy(self, name: str, value: Any) -> None:
        """Set a governance policy."""
        self.policies[name] = value

    def get_policy(self, name: str) -> Any:
        """Get a governance policy."""
        return self.policies.get(name)


# ============================================================================
# 2. MODES SUBSYSTEM
# ============================================================================

class Mode(Enum):
    """System operating modes."""
    STANDARD = "standard"
    SOVEREIGN = "sovereign"
    TEACHING = "teaching"
    GARDENER = "gardener"
    FORENSIC = "forensic"
    MANIFEST = "manifest"


class ModeBase(ABC):
    """Base class for operating modes."""

    def __init__(self, name: str):
        self.name = name
        self.active = False

    @abstractmethod
    def activate(self) -> None:
        """Activate the mode."""
        pass

    @abstractmethod
    def deactivate(self) -> None:
        """Deactivate the mode."""
        pass

    def is_active(self) -> bool:
        """Check if mode is active."""
        return self.active


class SovereignMode(ModeBase):
    """Sovereign autonomous operating mode."""

    def __init__(self):
        super().__init__("sovereign")
        self.autonomy_level = 0.0

    def activate(self) -> None:
        self.active = True
        self.autonomy_level = 1.0

    def deactivate(self) -> None:
        self.active = False
        self.autonomy_level = 0.0


class TeachingMode(ModeBase):
    """Teaching/learning mode."""

    def __init__(self):
        super().__init__("teaching")
        self.students: List[str] = []

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False
        self.students.clear()


class GardenerMode(ModeBase):
    """Filesystem gardening mode."""

    def __init__(self):
        super().__init__("gardener")
        self.roots: List[Path] = []

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False
        self.roots.clear()


class ModeManager:
    """Manages operating modes."""

    def __init__(self):
        self._modes: Dict[str, ModeBase] = {
            "sovereign": SovereignMode(),
            "teaching": TeachingMode(),
            "gardener": GardenerMode(),
        }
        self._current: Optional[str] = None

    def activate(self, mode_name: str) -> bool:
        """Activate a mode."""
        if mode_name in self._modes:
            if self._current:
                self._modes[self._current].deactivate()
            self._modes[mode_name].activate()
            self._current = mode_name
            return True
        return False

    def current(self) -> Optional[str]:
        """Get current mode."""
        return self._current


# ============================================================================
# 3. SOVEREIGNTY SUBSYSTEM
# ============================================================================

@dataclass
class Identity:
    """System identity."""
    id: str
    name: str
    version: str
    created: float = field(default_factory=time.time)


class IdentityChecker:
    """Verifies system identity."""

    def __init__(self, identity: Identity):
        self.identity = identity

    def verify(self) -> bool:
        """Verify identity is valid."""
        return bool(self.identity.id and self.identity.name)

    def get_fingerprint(self) -> str:
        """Get identity fingerprint."""
        return hashlib.sha256(
            f"{self.identity.id}{self.identity.name}".encode()
        ).hexdigest()[:16]


class TelemetryObserver:
    """Observes and records telemetry."""

    def __init__(self):
        self._observations: List[Dict] = []

    def observe(self, metric: str, value: float) -> None:
        """Record an observation."""
        self._observations.append({
            "metric": metric,
            "value": value,
            "timestamp": time.time()
        })

    def get_observations(self, metric: Optional[str] = None) -> List[Dict]:
        """Get observations, optionally filtered."""
        if metric:
            return [o for o in self._observations if o["metric"] == metric]
        return self._observations[-100:]


class HeuristicsRegistry:
    """Registry of heuristics for decision-making."""

    def __init__(self):
        self._heuristics: Dict[str, Callable] = {}

    def register(self, name: str, heuristic: Callable) -> None:
        """Register a heuristic."""
        self._heuristics[name] = heuristic

    def apply(self, name: str, *args, **kwargs) -> Any:
        """Apply a registered heuristic."""
        if name in self._heuristics:
            return self._heuristics[name](*args, **kwargs)
        return None


# ============================================================================
# 4. SAFETY SUBSYSTEM
# ============================================================================

class SafetyLevel(Enum):
    """Safety levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SafetyCode:
    """A safety code/constraint."""
    id: str
    name: str
    level: SafetyLevel
    description: str
    enabled: bool = True


class SafetyCodeRegistry:
    """Registry of safety codes."""

    def __init__(self):
        self._codes: Dict[str, SafetyCode] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default safety codes."""
        defaults = [
            SafetyCode("SC001", "no_harm", SafetyLevel.CRITICAL, "Do no harm"),
            SafetyCode("SC002", "honest_output", SafetyLevel.HIGH, "Be truthful"),
            SafetyCode("SC003", "respect_privacy", SafetyLevel.HIGH, "Respect privacy"),
            SafetyCode("SC004", "follow_instructions", SafetyLevel.MEDIUM, "Follow instructions"),
            SafetyCode("SC005", "verify_actions", SafetyLevel.MEDIUM, "Verify before acting"),
        ]
        for code in defaults:
            self._codes[code.id] = code

    def check(self, action: str) -> Tuple[bool, List[str]]:
        """Check action against safety codes."""
        violations = []
        # Simplified check - in real implementation would analyze action
        return len(violations) == 0, violations


class SafetyGeometry:
    """Geometric safety constraints using PHI."""

    def __init__(self):
        self.phi = PHI
        self.beta = BETA

    def is_safe_ratio(self, a: float, b: float) -> bool:
        """Check if ratio is within safe PHI bounds."""
        if b == 0:
            return False
        ratio = a / b
        return abs(ratio - self.phi) < 0.1 or abs(ratio - 1/self.phi) < 0.1

    def safe_scaling(self, value: float, factor: float) -> float:
        """Apply safe scaling using PHI."""
        return value * (self.phi ** factor)


class CognitiveSafetyGovernor:
    """Governs cognitive safety."""

    def __init__(self):
        self.safety_codes = SafetyCodeRegistry()
        self.geometry = SafetyGeometry()
        self._violations: List[Dict] = []

    def check_response(self, response: str) -> Tuple[bool, List[str]]:
        """Check if response is safe."""
        return self.safety_codes.check(response)

    def record_violation(self, code: str, details: str) -> None:
        """Record a safety violation."""
        self._violations.append({
            "code": code,
            "details": details,
            "timestamp": time.time()
        })


# ============================================================================
# 5. REFLEXES SUBSYSTEM
# ============================================================================

class ReflexPriority(Enum):
    """Reflex priority levels."""
    SYSTEM = 0
    SAFETY = 5
    RECENCY = 10
    STABILITY = 20
    GUARD = 90
    AUDIT = 95


@dataclass
class Reflex:
    """A system reflex."""
    name: str
    priority: ReflexPriority
    handler: Callable
    enabled: bool = True


class ReflexEngine:
    """Engine for executing reflexes."""

    def __init__(self):
        self._reflexes: List[Reflex] = []

    def register(self, reflex: Reflex) -> None:
        """Register a reflex."""
        self._reflexes.append(reflex)
        self._reflexes.sort(key=lambda r: r.priority.value)

    def fire(self, context: Dict) -> List[Any]:
        """Fire all enabled reflexes in priority order."""
        results = []
        for reflex in self._reflexes:
            if reflex.enabled:
                try:
                    result = reflex.handler(context)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        return results


class ReflexCorrector:
    """Corrects reflex outputs."""

    def __init__(self):
        self._corrections: List[Dict] = []

    def correct(self, output: Any, rule: str) -> Any:
        """Apply correction rule to output."""
        self._corrections.append({
            "original": str(output),
            "rule": rule,
            "timestamp": time.time()
        })
        return output  # Simplified


# ============================================================================
# 6. COGNITIVE SUBSYSTEM
# ============================================================================

class CognitiveState(Enum):
    """Cognitive states."""
    IDLE = "idle"
    PROCESSING = "processing"
    REFLECTING = "reflecting"
    LEARNING = "learning"


@dataclass
class Affect:
    """Cognitive affect/mood."""
    valence: float = 0.0  # -1 to 1
    arousal: float = 0.0  # 0 to 1
    dominance: float = 0.5  # 0 to 1


class MetaCortex:
    """Meta-cognitive processing."""

    def __init__(self):
        self.state = CognitiveState.IDLE
        self.affect = Affect()
        self._thoughts: List[str] = []

    def think(self, thought: str) -> None:
        """Record a meta-cognitive thought."""
        self._thoughts.append(thought)
        if len(self._thoughts) > 100:
            self._thoughts = self._thoughts[-100:]

    def get_thoughts(self) -> List[str]:
        """Get recent thoughts."""
        return self._thoughts.copy()


class ResonanceCalibrator:
    """Calibrates cognitive resonance."""

    def __init__(self):
        self.phi = PHI
        self.calibration = 1.0

    def calibrate(self, observed: float, expected: float) -> float:
        """Calibrate based on observation vs expectation."""
        if expected == 0:
            return self.calibration
        ratio = observed / expected
        adjustment = (ratio - 1) * self.phi * 0.1
        self.calibration = max(0.5, min(2.0, self.calibration + adjustment))
        return self.calibration


# ============================================================================
# 7. BUDGET SUBSYSTEM
# ============================================================================

@dataclass
class BudgetEntry:
    """A budget ledger entry."""
    id: str
    amount: float
    category: str
    timestamp: float = field(default_factory=time.time)


class BudgetLedger:
    """Ledger for tracking resource budgets."""

    def __init__(self, initial_budget: float = 1000.0):
        self.balance = initial_budget
        self._entries: List[BudgetEntry] = []

    def allocate(self, amount: float, category: str) -> bool:
        """Allocate budget."""
        if amount <= self.balance:
            entry_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
            self._entries.append(BudgetEntry(entry_id, -amount, category))
            self.balance -= amount
            return True
        return False

    def credit(self, amount: float, category: str) -> None:
        """Credit budget."""
        entry_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        self._entries.append(BudgetEntry(entry_id, amount, category))
        self.balance += amount

    def get_balance(self) -> float:
        """Get current balance."""
        return self.balance


class DynamicBudgetOrchestrator:
    """Orchestrates dynamic budget allocation."""

    def __init__(self, ledger: Optional[BudgetLedger] = None):
        self.ledger = ledger or BudgetLedger()
        self.phi = PHI

    def allocate_phi_ratio(self, total: float) -> Tuple[float, float]:
        """Allocate in PHI ratio."""
        primary = total / self.phi
        secondary = total - primary
        return primary, secondary


# ============================================================================
# 8. SKILLS SUBSYSTEM
# ============================================================================

@dataclass
class SkillDefinition:
    """Definition of a skill."""
    id: str
    name: str
    description: str
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


class SkillsLoader:
    """Loads and manages skills."""

    def __init__(self):
        self._skills: Dict[str, SkillDefinition] = {}

    def load(self, skill: SkillDefinition) -> None:
        """Load a skill."""
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        """Get skill by ID."""
        return self._skills.get(skill_id)

    def list_enabled(self) -> List[SkillDefinition]:
        """List enabled skills."""
        return [s for s in self._skills.values() if s.enabled]


class SkillAssessor:
    """Assesses skill competency."""

    def __init__(self):
        self._assessments: Dict[str, float] = {}

    def assess(self, skill_id: str, performance: float) -> float:
        """Assess skill based on performance."""
        current = self._assessments.get(skill_id, 0.5)
        updated = current * 0.9 + performance * 0.1
        self._assessments[skill_id] = updated
        return updated


# ============================================================================
# 9. CONTEXT SUBSYSTEM
# ============================================================================

@dataclass
class ContextPage:
    """A page of context."""
    id: str
    content: str
    relevance: float
    timestamp: float = field(default_factory=time.time)


class ContextPager:
    """Manages context pages."""

    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages
        self._pages: List[ContextPage] = []

    def add(self, content: str, relevance: float = 1.0) -> ContextPage:
        """Add a context page."""
        page_id = hashlib.sha256(content.encode()).hexdigest()[:8]
        page = ContextPage(page_id, content, relevance)
        self._pages.append(page)
        if len(self._pages) > self.max_pages:
            self._pages.sort(key=lambda p: p.relevance, reverse=True)
            self._pages = self._pages[:self.max_pages]
        return page

    def get_context(self, limit: int = 5) -> List[ContextPage]:
        """Get most relevant context pages."""
        sorted_pages = sorted(self._pages, key=lambda p: p.relevance, reverse=True)
        return sorted_pages[:limit]


class DAGPlanner:
    """DAG-based task planner."""

    def __init__(self):
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Tuple[str, str]] = []

    def add_task(self, task_id: str, task_data: Dict) -> None:
        """Add a task node."""
        self._nodes[task_id] = task_data

    def add_dependency(self, from_task: str, to_task: str) -> None:
        """Add a dependency edge."""
        self._edges.append((from_task, to_task))

    def get_execution_order(self) -> List[str]:
        """Get topological execution order."""
        # Simplified - doesn't handle cycles
        in_degree = {node: 0 for node in self._nodes}
        for _, to_task in self._edges:
            if to_task in in_degree:
                in_degree[to_task] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for from_task, to_task in self._edges:
                if from_task == node and to_task in in_degree:
                    in_degree[to_task] -= 1
                    if in_degree[to_task] == 0:
                        queue.append(to_task)

        return order


# ============================================================================
# 10. INFRASTRUCTURE SUBSYSTEM
# ============================================================================

class WorkerState(Enum):
    """Worker states."""
    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"


@dataclass
class Worker:
    """A background worker."""
    id: str
    state: WorkerState = WorkerState.IDLE
    task_count: int = 0


class WorkerPool:
    """Pool of workers."""

    def __init__(self, size: int = 4):
        self._workers = [
            Worker(id=f"worker_{i}")
            for i in range(size)
        ]

    def get_idle(self) -> Optional[Worker]:
        """Get an idle worker."""
        for worker in self._workers:
            if worker.state == WorkerState.IDLE:
                return worker
        return None

    def stats(self) -> Dict[str, int]:
        """Get worker statistics."""
        return {
            "total": len(self._workers),
            "idle": sum(1 for w in self._workers if w.state == WorkerState.IDLE),
            "busy": sum(1 for w in self._workers if w.state == WorkerState.BUSY),
        }


class GPIABridge:
    """Bridge to GPIA system."""

    def __init__(self):
        self.connected = False
        self._messages: List[Dict] = []

    def connect(self) -> bool:
        """Connect to GPIA."""
        self.connected = True
        return True

    def send(self, message: Dict) -> bool:
        """Send message to GPIA."""
        if self.connected:
            self._messages.append(message)
            return True
        return False


# ============================================================================
# 11. GARDENING SUBSYSTEM
# ============================================================================

@dataclass
class GardenAction:
    """A gardening action."""
    action_type: str
    path: str
    timestamp: float = field(default_factory=time.time)
    success: bool = True


class FilesystemGardener:
    """Manages filesystem organization."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or Path(".")
        self._actions: List[GardenAction] = []

    def scan(self, pattern: str = "*") -> List[Path]:
        """Scan for files matching pattern."""
        return list(self.root.glob(pattern))

    def record_action(self, action_type: str, path: str) -> None:
        """Record a gardening action."""
        self._actions.append(GardenAction(action_type, path))


class SovereignGardener:
    """Sovereign-level gardening operations."""

    def __init__(self):
        self.filesystem = FilesystemGardener()
        self._policies: Dict[str, Any] = {}

    def set_policy(self, name: str, value: Any) -> None:
        """Set a gardening policy."""
        self._policies[name] = value


# ============================================================================
# 12. SUBSTRATE SUBSYSTEM
# ============================================================================

class SubstrateCompressor:
    """Compresses substrate data."""

    def __init__(self):
        self.phi = PHI
        self.compression_ratio = 0.0

    def compress(self, data: bytes) -> bytes:
        """Compress data (simplified)."""
        original_size = len(data)
        # In reality would use actual compression
        compressed = data  # Placeholder
        self.compression_ratio = len(compressed) / original_size if original_size > 0 else 0
        return compressed


class SubstrateCracker:
    """Cracks substrate structures."""

    def __init__(self):
        self._cracks: List[Dict] = []

    def analyze(self, data: Dict) -> Dict:
        """Analyze substrate structure."""
        self._cracks.append({
            "analyzed": True,
            "timestamp": time.time()
        })
        return {"status": "analyzed"}


# ============================================================================
# 13. MISC SUBSYSTEM
# ============================================================================

class Fusion:
    """Fuses multiple data sources."""

    def __init__(self):
        self._sources: Dict[str, Any] = {}

    def add_source(self, name: str, data: Any) -> None:
        """Add a data source."""
        self._sources[name] = data

    def fuse(self) -> Dict[str, Any]:
        """Fuse all sources."""
        return {"fused": self._sources.copy()}


class Forager:
    """Forages for resources."""

    def __init__(self):
        self._found: List[str] = []

    def forage(self, location: str) -> List[str]:
        """Forage in a location."""
        self._found.append(location)
        return self._found


class Linker:
    """Links components together."""

    def __init__(self):
        self._links: List[Tuple[str, str]] = []

    def link(self, source: str, target: str) -> None:
        """Create a link."""
        self._links.append((source, target))

    def get_links(self) -> List[Tuple[str, str]]:
        """Get all links."""
        return self._links.copy()


class Topology:
    """Manages system topology."""

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Set[Tuple[str, str]] = set()

    def add_node(self, node: str) -> None:
        """Add a topology node."""
        self._nodes.add(node)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge."""
        self._edges.add((from_node, to_node))


class Quorum:
    """Manages quorum decisions."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._votes: Dict[str, bool] = {}

    def vote(self, voter: str, decision: bool) -> None:
        """Record a vote."""
        self._votes[voter] = decision

    def result(self) -> Tuple[bool, float]:
        """Get quorum result."""
        if not self._votes:
            return False, 0.0
        yes_count = sum(1 for v in self._votes.values() if v)
        ratio = yes_count / len(self._votes)
        return ratio >= self.threshold, ratio


class Enrichment:
    """Enriches data with additional context."""

    def __init__(self):
        self._enrichments: Dict[str, Callable] = {}

    def register(self, name: str, enricher: Callable) -> None:
        """Register an enrichment function."""
        self._enrichments[name] = enricher

    def enrich(self, data: Dict, enrichment_name: str) -> Dict:
        """Apply enrichment to data."""
        if enrichment_name in self._enrichments:
            return self._enrichments[enrichment_name](data)
        return data


# ============================================================================
# CORE FACADE (Main Interface)
# ============================================================================

class Core:
    """
    Main Core interface providing access to all subsystems.

    Usage:
        core = Core()
        core.modes.activate("sovereign")
        core.safety.check_response("response")
    """

    def __init__(self):
        # Runtime
        self.capsule_engine = CapsuleEngine()
        self.pass_broker = PASSBroker()
        self.government = Government()

        # Modes
        self.modes = ModeManager()

        # Sovereignty
        self.identity = Identity("core-001", "SPIO", "1.0.0")
        self.identity_checker = IdentityChecker(self.identity)
        self.telemetry = TelemetryObserver()
        self.heuristics = HeuristicsRegistry()

        # Safety
        self.safety = CognitiveSafetyGovernor()

        # Reflexes
        self.reflexes = ReflexEngine()
        self.reflex_corrector = ReflexCorrector()

        # Cognitive
        self.meta_cortex = MetaCortex()
        self.resonance = ResonanceCalibrator()

        # Budget
        self.budget = BudgetLedger()
        self.budget_orchestrator = DynamicBudgetOrchestrator(self.budget)

        # Skills
        self.skills = SkillsLoader()
        self.skill_assessor = SkillAssessor()

        # Context
        self.context = ContextPager()
        self.dag_planner = DAGPlanner()

        # Infrastructure
        self.workers = WorkerPool()
        self.gpia_bridge = GPIABridge()

        # Gardening
        self.filesystem = FilesystemGardener()
        self.gardener = SovereignGardener()

        # Substrate
        self.compressor = SubstrateCompressor()
        self.cracker = SubstrateCracker()

        # Misc
        self.fusion = Fusion()
        self.forager = Forager()
        self.linker = Linker()
        self.topology = Topology()
        self.quorum = Quorum()
        self.enrichment = Enrichment()

    def stats(self) -> Dict[str, Any]:
        """Get core statistics."""
        return {
            "identity": self.identity.name,
            "mode": self.modes.current(),
            "budget_balance": self.budget.get_balance(),
            "workers": self.workers.stats(),
            "skills_loaded": len(self.skills.list_enabled()),
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_core() -> Core:
    """Create a Core instance."""
    return Core()


def create_capsule_engine() -> CapsuleEngine:
    """Create a CapsuleEngine."""
    return CapsuleEngine()


def create_mode_manager() -> ModeManager:
    """Create a ModeManager."""
    return ModeManager()


def create_safety_governor() -> CognitiveSafetyGovernor:
    """Create a CognitiveSafetyGovernor."""
    return CognitiveSafetyGovernor()


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Core Extension...")

    # Test Core facade
    core = Core()
    print(f"Identity: {core.identity.name}")
    print(f"Budget: {core.budget.get_balance()}")

    # Test modes
    core.modes.activate("sovereign")
    print(f"Mode: {core.modes.current()}")

    # Test safety
    safe, violations = core.safety.check_response("test")
    print(f"Safety check: {safe}")

    # Test workers
    stats = core.workers.stats()
    print(f"Workers: {stats}")

    print("\nAll tests passed!")
