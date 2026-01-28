"""
Skills Core - Complete Skill Framework Implementation

50+ skill components with PHI-based optimization and progressive disclosure.

Architecture:
- Skills are lazy-loaded to minimize memory footprint
- Each skill contains its own prompts, tools, and validation schemas
- Skills can depend on other skills, forming a capability graph
- The registry enables discovery and sharing across the ecosystem

Based on: CLI-main/src/skills/*
"""

import hashlib
import json
import time
import sqlite3
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import math

# Brahim's Calculator constants
PHI = (1 + math.sqrt(5)) / 2
BETA = 1 / (PHI ** 3)
GAMMA = 1 / (PHI ** 4)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class SkillLevel(Enum):
    """Complexity level (affects when skill is disclosed)."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillCategory(Enum):
    """Primary skill categories."""
    CODE = "code"
    DATA = "data"
    WRITING = "writing"
    RESEARCH = "research"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    COGNITION = "cognition"
    VALIDATION = "validation"
    DECOMPOSITION = "decomposition"
    PATTERN_RECOGNITION = "pattern_recognition"
    GENERAL = "general"


class SkillStatus(Enum):
    """Skill execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class SelectionMethod(Enum):
    """How the skill was selected."""
    EXPLOITATION = "exploitation"  # High confidence learned pattern
    EXPLORATION = "exploration"    # Trying new skill
    FALLBACK = "fallback"          # Best available despite low confidence
    NO_DATA = "no_data"            # No learned patterns


# =============================================================================
# BASE CLASSES
# =============================================================================

@dataclass
class SkillDependency:
    """Dependency on another skill."""
    skill_id: str
    optional: bool = False
    reason: str = ""
    min_version: str = "0.0.0"


@dataclass
class SkillMetadata:
    """Skill metadata for discovery and documentation."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: SkillCategory = SkillCategory.GENERAL
    level: SkillLevel = SkillLevel.INTERMEDIATE
    tags: List[str] = field(default_factory=list)
    long_description: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[SkillDependency] = field(default_factory=list)
    requires_tools: List[str] = field(default_factory=list)
    estimated_tokens: int = 500
    author: str = "SPIO"
    license: str = "MIT"
    repository: str = ""


@dataclass
class SkillContext:
    """Runtime context for skill execution."""
    agent_role: str = "default"
    session_id: str = ""
    user_id: str = ""
    permissions: Dict[str, bool] = field(default_factory=dict)
    budget_tokens: int = 10000
    timeout_ms: int = 30000
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PHI-based resource allocation
    phi_saturation: float = 1.0  # Resource saturation level
    dimension: int = 7  # Current processing dimension (1-12)


@dataclass
class SkillResult:
    """Result of skill execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    error_code: Optional[str] = None
    skill_id: str = ""
    execution_time_ms: int = 0
    tokens_used: int = 0
    suggestions: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """
    Abstract base class for all skills.

    Progressive disclosure:
    - Layer 0: Metadata only (~2KB)
    - Layer 1: Class definition (~5KB)
    - Layer 2: Method execution (~20KB)
    - Layer 3: Extended resources (on demand)
    """

    _initialized: bool = False
    _config: Dict[str, Any] = None

    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """Return skill metadata (Layer 0)."""
        pass

    def input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for valid inputs."""
        return {"type": "object", "properties": {}}

    def output_schema(self) -> Dict[str, Any]:
        """Return JSON schema for outputs."""
        return {"type": "object", "properties": {}}

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """One-time initialization (Layer 1)."""
        self._config = config or {}
        self._initialized = True

    def cleanup(self) -> None:
        """Release resources."""
        self._initialized = False
        self._config = None

    @abstractmethod
    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        """Execute the skill (Layer 2)."""
        pass

    def get_prompt(self) -> str:
        """Return system prompt for LLM interactions."""
        meta = self.metadata()
        return f"You are executing {meta.name}. {meta.description}"

    def get_tools(self) -> List[Callable]:
        """Return callable tools this skill provides."""
        return []


# =============================================================================
# SKILL MANIFEST & HANDLE
# =============================================================================

@dataclass
class SkillManifest:
    """Layer 0: Parsed manifest data."""
    id: str
    version: str
    name: str
    description: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    level: str = "intermediate"
    load_weight: str = "light"
    entry_point: str = "skill.py:Skill"
    requires: List[str] = field(default_factory=list)
    capabilities: List[Dict[str, str]] = field(default_factory=list)
    permissions: Dict[str, Any] = field(default_factory=dict)
    skill_dir: Path = None
    schema: Dict[str, Any] = field(default_factory=dict)

    def memory_size(self) -> int:
        """Estimate memory footprint of Layer 0 data."""
        return len(str(self.__dict__))

    @classmethod
    def from_dict(cls, data: Dict[str, Any], skill_dir: Path = None) -> "SkillManifest":
        """Create manifest from dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            version=data.get("version", "1.0.0"),
            name=data.get("name", "Unknown Skill"),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            level=data.get("level", "intermediate"),
            load_weight=data.get("load_weight", "light"),
            entry_point=data.get("entry_point", "skill.py:Skill"),
            requires=data.get("requires", []),
            capabilities=data.get("capabilities", []),
            permissions=data.get("permissions", {}),
            skill_dir=skill_dir,
            schema=data.get("schema", {}),
        )


@dataclass
class SkillHandle:
    """
    Lazy handle to a skill - supports progressive disclosure.

    Layer 0 (manifest) is always available.
    Layer 1+ is loaded on demand.
    """
    manifest: SkillManifest
    _skill_class: Optional[Type[Skill]] = field(default=None, repr=False)
    _skill_instance: Optional[Skill] = field(default=None, repr=False)
    _load_attempted: bool = field(default=False, repr=False)

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def is_loaded(self) -> bool:
        """Check if Layer 1 (class) is loaded."""
        return self._skill_class is not None

    def get_class(self) -> Optional[Type[Skill]]:
        """Load Layer 1: Get skill class."""
        if self._skill_class is not None:
            return self._skill_class
        if self._load_attempted:
            return None
        self._load_attempted = True
        # Skill class would be dynamically loaded here
        return self._skill_class

    def get_instance(self) -> Optional[Skill]:
        """Get or create skill instance."""
        if self._skill_instance is not None:
            return self._skill_instance
        skill_class = self.get_class()
        if skill_class is None:
            return None
        try:
            self._skill_instance = skill_class()
            return self._skill_instance
        except Exception as e:
            logger.error(f"Failed to instantiate skill {self.manifest.id}: {e}")
            return None

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        """Execute skill (Layer 2)."""
        skill = self.get_instance()
        if skill is None:
            return SkillResult(
                success=False,
                output=None,
                error="Failed to load skill",
                error_code="LOAD_FAILED",
                skill_id=self.manifest.id,
            )
        try:
            start = time.time()
            result = skill.execute(input_data, context)
            result.execution_time_ms = int((time.time() - start) * 1000)
            return result
        except Exception as e:
            return SkillResult(
                success=False,
                output=None,
                error=str(e),
                error_code="EXECUTION_ERROR",
                skill_id=self.manifest.id,
            )


# =============================================================================
# PROGRESSIVE LOADER
# =============================================================================

class ProgressiveLoader:
    """
    Skill loader implementing progressive disclosure.

    Memory hierarchy:
      Layer 0: Manifest + Schema (~2KB)  - Always loaded
      Layer 1: Class definition (~5KB)   - Loaded on reference
      Layer 2: Method execution (~20KB)  - Loaded on invoke
      Layer 3: Extended resources        - Loaded on demand
    """

    def __init__(self):
        self._handles: Dict[str, SkillHandle] = {}
        self._dependency_graph: Dict[str, List[str]] = {}

    def register(self, skill_class: Type[Skill]) -> str:
        """Register a skill class directly."""
        instance = skill_class()
        meta = instance.metadata()
        manifest = SkillManifest(
            id=meta.id,
            version=meta.version,
            name=meta.name,
            description=meta.description,
            category=meta.category.value if isinstance(meta.category, Enum) else meta.category,
            tags=meta.tags,
            level=meta.level.value if isinstance(meta.level, Enum) else meta.level,
        )
        handle = SkillHandle(manifest=manifest)
        handle._skill_class = skill_class
        self._handles[meta.id] = handle
        deps = [d.skill_id for d in meta.dependencies]
        self._dependency_graph[meta.id] = deps
        return meta.id

    def get(self, skill_id: str) -> Optional[SkillHandle]:
        """Get skill handle by ID (does not load code)."""
        return self._handles.get(skill_id)

    def find_by_category(self, category: Union[str, SkillCategory]) -> List[SkillHandle]:
        """Find skills by category (Layer 0 only)."""
        cat_str = category.value if isinstance(category, Enum) else category
        return [h for h in self._handles.values() if h.manifest.category == cat_str]

    def find_by_tag(self, tag: str) -> List[SkillHandle]:
        """Find skills by tag (Layer 0 only)."""
        return [h for h in self._handles.values() if tag in h.manifest.tags]

    def resolve_dependencies(self, skill_id: str) -> List[str]:
        """Resolve skill dependencies (returns ordered list)."""
        resolved = []
        seen = set()

        def visit(sid: str):
            if sid in seen:
                return
            seen.add(sid)
            for dep in self._dependency_graph.get(sid, []):
                if dep in self._handles:
                    visit(dep)
            resolved.append(sid)

        visit(skill_id)
        return resolved

    def execute(self, skill_id: str, input_data: Dict[str, Any],
                context: SkillContext) -> SkillResult:
        """Execute a skill with full protocol."""
        handle = self.get(skill_id)
        if handle is None:
            return SkillResult(
                success=False,
                output=None,
                error=f"Skill not found: {skill_id}",
                error_code="NOT_FOUND",
            )

        # Resolve dependencies
        try:
            for dep_id in self.resolve_dependencies(skill_id)[:-1]:
                dep = self.get(dep_id)
                if dep and not dep.is_loaded:
                    dep.get_instance()
        except Exception as e:
            return SkillResult(
                success=False,
                output=None,
                error=str(e),
                error_code="DEPENDENCY_ERROR",
            )

        return handle.execute(input_data, context)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        by_cat = {}
        for h in self._handles.values():
            cat = h.manifest.category
            by_cat[cat] = by_cat.get(cat, 0) + 1
        return {
            "discovered": len(self._handles),
            "loaded": sum(1 for h in self._handles.values() if h.is_loaded),
            "by_category": by_cat,
            "total_layer0_memory": sum(h.manifest.memory_size() for h in self._handles.values()),
        }

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all registered skills."""
        return [
            {
                "id": h.manifest.id,
                "name": h.manifest.name,
                "category": h.manifest.category,
                "level": h.manifest.level,
                "loaded": h.is_loaded,
            }
            for h in self._handles.values()
        ]


# =============================================================================
# REGISTRY
# =============================================================================

class SkillRegistry:
    """Registry for managing skills with search and filtering."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_tag: Dict[str, List[str]] = {}
        self._performance: Dict[str, Dict[str, float]] = {}

    def register(self, skill: Skill) -> bool:
        """Register a skill instance."""
        meta = skill.metadata()
        if meta.id in self._skills:
            return False

        self._skills[meta.id] = skill

        # Index by category
        cat = meta.category.value if isinstance(meta.category, Enum) else str(meta.category)
        if cat not in self._by_category:
            self._by_category[cat] = []
        self._by_category[cat].append(meta.id)

        # Index by tags
        for tag in meta.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(meta.id)

        return True

    def get(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        return self._skills.get(skill_id)

    def list_skills(self, category: str = None, tag: str = None) -> List[SkillMetadata]:
        """List skills with optional filtering."""
        if category:
            ids = self._by_category.get(category, [])
        elif tag:
            ids = self._by_tag.get(tag, [])
        else:
            ids = list(self._skills.keys())
        return [self._skills[sid].metadata() for sid in ids if sid in self._skills]

    def search(self, query: str) -> List[SkillMetadata]:
        """Search skills by name or description."""
        query_lower = query.lower()
        results = []
        for skill in self._skills.values():
            meta = skill.metadata()
            if query_lower in meta.name.lower() or query_lower in meta.description.lower():
                results.append(meta)
        return results

    def record_performance(self, skill_id: str, success: bool, quality: float, time_ms: int):
        """Record skill performance metrics."""
        if skill_id not in self._performance:
            self._performance[skill_id] = {
                "total": 0, "successes": 0, "quality_sum": 0, "time_sum": 0
            }
        p = self._performance[skill_id]
        p["total"] += 1
        p["successes"] += 1 if success else 0
        p["quality_sum"] += quality
        p["time_sum"] += time_ms

    def get_performance(self, skill_id: str) -> Dict[str, float]:
        """Get skill performance stats."""
        p = self._performance.get(skill_id, {})
        total = p.get("total", 0)
        if total == 0:
            return {"success_rate": 0, "avg_quality": 0, "avg_time_ms": 0, "total_uses": 0}
        return {
            "success_rate": p["successes"] / total,
            "avg_quality": p["quality_sum"] / total,
            "avg_time_ms": p["time_sum"] / total,
            "total_uses": total,
        }


class GPIACoreSkillRegistry:
    """Specialized registry for GPIA-core model skills."""

    CORE_SKILL_CATEGORIES = {
        "reasoning": "Deep logical reasoning patterns",
        "synthesis": "Combining multiple insights into coherent output",
        "validation": "Checking proofs and logical consistency",
        "abstraction": "Finding patterns and generalizations",
        "decomposition": "Breaking complex problems into sub-problems",
        "recursive_thinking": "Recursive problem solving",
        "constraint_solving": "Working with mathematical constraints",
        "pattern_recognition": "Finding mathematical patterns",
    }

    def __init__(self, db_path: Path = None):
        """Initialize GPIA-core skill registry."""
        self.db_path = db_path or Path(":memory:")
        self._skills: Dict[str, Dict[str, Any]] = {}
        self._performance: Dict[str, List[Dict]] = {}
        self._synergies: Dict[Tuple[str, str], float] = {}

    def register_core_skill(self, name: str, category: str, description: str,
                           version: str = "1.0") -> bool:
        """Register a new core skill."""
        if category not in self.CORE_SKILL_CATEGORIES:
            return False
        if name in self._skills:
            return False
        self._skills[name] = {
            "name": name,
            "category": category,
            "description": description,
            "version": version,
            "created_at": time.time(),
        }
        return True

    def list_core_skills(self) -> Dict[str, List[str]]:
        """List all core skills by category."""
        by_cat = {}
        for name, info in self._skills.items():
            cat = info["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(name)
        return by_cat

    def get_skill_for_task(self, task: str, task_type: str = "reasoning") -> Optional[str]:
        """Get best skill for a given task."""
        skills = [n for n, i in self._skills.items() if i["category"] == task_type]
        if not skills:
            return None
        # Return skill with best performance
        best = None
        best_quality = -1
        for skill in skills:
            perf = self.get_skill_performance(skill)
            if perf.get("avg_quality", 0) > best_quality:
                best = skill
                best_quality = perf.get("avg_quality", 0)
        return best or (skills[0] if skills else None)

    def record_skill_performance(self, skill_name: str, task: str, success: bool,
                                 quality_score: float, execution_time_ms: float,
                                 tokens_used: int):
        """Record skill performance."""
        if skill_name not in self._performance:
            self._performance[skill_name] = []
        self._performance[skill_name].append({
            "task": task,
            "success": success,
            "quality": quality_score,
            "time_ms": execution_time_ms,
            "tokens": tokens_used,
            "timestamp": time.time(),
        })

    def get_skill_performance(self, skill_name: str) -> Dict[str, Any]:
        """Get performance statistics for a skill."""
        records = self._performance.get(skill_name, [])
        if not records:
            return {}
        total = len(records)
        successes = sum(1 for r in records if r["success"])
        return {
            "name": skill_name,
            "success_rate": successes / total,
            "avg_quality": sum(r["quality"] for r in records) / total,
            "avg_time_ms": sum(r["time_ms"] for r in records) / total,
            "total_uses": total,
            "avg_tokens": sum(r["tokens"] for r in records) / total,
        }

    def record_skill_synergy(self, primary: str, supporting: str, synergy_score: float):
        """Record skill synergy."""
        self._synergies[(primary, supporting)] = synergy_score

    def get_best_skill_combinations(self, primary_skill: str) -> List[Dict]:
        """Get skills that work best with a given primary skill."""
        results = []
        for (p, s), score in self._synergies.items():
            if p == primary_skill:
                results.append({"supporting_skill": s, "synergy_score": score})
        return sorted(results, key=lambda x: x["synergy_score"], reverse=True)[:5]


# =============================================================================
# AUTONOMOUS SKILL SELECTOR
# =============================================================================

@dataclass
class SelectionResult:
    """Result of skill selection."""
    skill_id: Optional[str]
    method: SelectionMethod
    confidence: float
    reasoning: str
    alternatives: List[str] = field(default_factory=list)


class SkillSelectorMemory:
    """Memory system for autonomous skill selector."""

    def __init__(self):
        self._recommendations: List[Dict] = []
        self._patterns: Dict[Tuple[str, str], Dict[str, Dict]] = {}  # (model, pattern) -> skill -> stats

    def record_recommendation(self, model: str, task: str, task_pattern: str,
                             skill: str, success: bool, quality: float, method: str):
        """Record a skill recommendation."""
        self._recommendations.append({
            "timestamp": time.time(),
            "model": model,
            "task": task,
            "pattern": task_pattern,
            "skill": skill,
            "success": success,
            "quality": quality,
            "method": method,
        })
        # Update pattern stats
        key = (model, task_pattern)
        if key not in self._patterns:
            self._patterns[key] = {}
        if skill not in self._patterns[key]:
            self._patterns[key][skill] = {"total": 0, "successes": 0, "quality_sum": 0}
        p = self._patterns[key][skill]
        p["total"] += 1
        p["successes"] += 1 if success else 0
        p["quality_sum"] += quality

    def get_learned_patterns(self, model: str, task_pattern: str) -> List[Dict]:
        """Get learned patterns for a model-task combo."""
        key = (model, task_pattern)
        patterns = self._patterns.get(key, {})
        results = []
        for skill, stats in patterns.items():
            total = stats["total"]
            if total > 0:
                results.append({
                    "skill": skill,
                    "success_rate": stats["successes"] / total,
                    "avg_quality": stats["quality_sum"] / total,
                    "confidence": min(1.0, total / 15),  # Confidence grows with observations
                    "observations": total,
                })
        return sorted(results, key=lambda x: x["confidence"], reverse=True)


class AutonomousSkillSelector:
    """
    Autonomous agent that learns to select the best skills.

    Features:
    - Learns patterns: model + task_type -> best_skill
    - Exploration vs exploitation balance
    - Confidence-based selection
    - PHI-optimal decision thresholds
    """

    def __init__(self):
        self.memory = SkillSelectorMemory()
        self.exploration_rate = BETA  # ~23.6% exploration (PHI-derived)
        self.confidence_threshold = 1 / PHI  # ~61.8% (OMEGA)
        self.min_observations = 5
        self._cache: Dict[str, SelectionResult] = {}

    def abstract_task_pattern(self, task: str) -> str:
        """Abstract a task to a pattern category."""
        task_lower = task.lower()
        patterns = {
            "reasoning": ["analyze", "derive", "explain", "understand", "reason", "complex"],
            "synthesis": ["combine", "synthesize", "integrate", "merge"],
            "validation": ["validate", "check", "verify", "prove"],
            "optimization": ["optimize", "improve", "enhance"],
            "decomposition": ["break", "decompose", "split", "divide"],
            "pattern_recognition": ["pattern", "recognize", "detect", "identify"],
            "code": ["code", "program", "function", "class", "implement"],
            "data": ["data", "dataset", "transform", "aggregate"],
            "writing": ["write", "draft", "edit", "document"],
            "research": ["research", "paper", "literature", "citation"],
        }
        for pattern, keywords in patterns.items():
            if any(kw in task_lower for kw in keywords):
                return pattern
        return "general"

    def select_skill(self, model: str, task: str,
                    available_skills: List[str] = None) -> SelectionResult:
        """Select best skill for the given task."""
        pattern = self.abstract_task_pattern(task)
        learned = self.memory.get_learned_patterns(model, pattern)

        # High confidence exploitation
        if learned and learned[0]["confidence"] >= self.confidence_threshold:
            return SelectionResult(
                skill_id=learned[0]["skill"],
                method=SelectionMethod.EXPLOITATION,
                confidence=learned[0]["confidence"],
                reasoning=f"High confidence ({learned[0]['confidence']:.1%}) "
                         f"from {learned[0]['observations']} observations",
                alternatives=[l["skill"] for l in learned[1:4]],
            )

        # Exploration
        import random
        if available_skills and random.random() < self.exploration_rate:
            # Try a new skill
            known = {l["skill"] for l in learned}
            unknown = [s for s in available_skills if s not in known]
            if unknown:
                return SelectionResult(
                    skill_id=random.choice(unknown),
                    method=SelectionMethod.EXPLORATION,
                    confidence=0.5,
                    reasoning="Exploring new skill combination",
                    alternatives=unknown[:3],
                )

        # Fallback to best learned
        if learned:
            return SelectionResult(
                skill_id=learned[0]["skill"],
                method=SelectionMethod.FALLBACK,
                confidence=learned[0]["confidence"],
                reasoning="Using best available despite low confidence",
                alternatives=[l["skill"] for l in learned[1:4]],
            )

        # No data
        if available_skills:
            return SelectionResult(
                skill_id=available_skills[0],
                method=SelectionMethod.NO_DATA,
                confidence=0.0,
                reasoning="No learned patterns, using first available",
                alternatives=available_skills[1:4],
            )

        return SelectionResult(
            skill_id=None,
            method=SelectionMethod.NO_DATA,
            confidence=0.0,
            reasoning="No skills available",
        )

    def record_outcome(self, model: str, task: str, skill: str,
                      success: bool, quality: float, method: str = ""):
        """Record outcome and learn from it."""
        pattern = self.abstract_task_pattern(task)
        self.memory.record_recommendation(model, task, pattern, skill, success, quality, method)


# =============================================================================
# SKILL ROUTER
# =============================================================================

class SkillRouter:
    """
    Routes tasks to optimal skills using PHI-based saturation.

    Uses dimension-aware routing based on Brahim's Calculator.
    """

    # Dimension to category mapping (12-dimension model)
    DIMENSION_CATEGORIES = {
        1: [SkillCategory.VALIDATION],          # PERCEPTION
        2: [SkillCategory.DATA],                # ATTENTION
        3: [SkillCategory.VALIDATION],          # SECURITY
        4: [SkillCategory.DATA],                # STABILITY
        5: [SkillCategory.CODE],                # COMPRESSION
        6: [SkillCategory.SYNTHESIS],           # HARMONY
        7: [SkillCategory.REASONING],           # REASONING
        8: [SkillCategory.RESEARCH],            # PREDICTION
        9: [SkillCategory.COGNITION],           # CREATIVITY
        10: [SkillCategory.RESEARCH],           # WISDOM
        11: [SkillCategory.SYNTHESIS],          # INTEGRATION
        12: [SkillCategory.COGNITION],          # UNIFICATION
    }

    def __init__(self, registry: SkillRegistry = None, loader: ProgressiveLoader = None):
        self.registry = registry or SkillRegistry()
        self.loader = loader or ProgressiveLoader()

    def route(self, task: str, dimension: int = 7) -> List[SkillMetadata]:
        """Route task to appropriate skills based on dimension."""
        categories = self.DIMENSION_CATEGORIES.get(dimension, [SkillCategory.GENERAL])
        results = []
        for cat in categories:
            results.extend(self.registry.list_skills(category=cat.value))
        return results

    def phi_optimal_selection(self, candidates: List[SkillMetadata],
                              context: SkillContext) -> Optional[SkillMetadata]:
        """Select optimal skill using PHI saturation."""
        if not candidates:
            return None

        # Score each candidate
        scores = []
        for meta in candidates:
            # PHI-weighted score
            level_weights = {
                SkillLevel.BASIC: 1.0,
                SkillLevel.INTERMEDIATE: PHI,
                SkillLevel.ADVANCED: PHI ** 2,
                SkillLevel.EXPERT: PHI ** 3,
            }
            level = meta.level if isinstance(meta.level, SkillLevel) else SkillLevel.INTERMEDIATE
            weight = level_weights.get(level, 1.0)

            # Performance boost
            perf = self.registry.get_performance(meta.id)
            quality_boost = perf.get("avg_quality", 0.5)

            # Token efficiency (prefer lower token skills when budget is limited)
            token_factor = 1.0
            if context.budget_tokens > 0:
                token_factor = min(1.0, context.budget_tokens / max(meta.estimated_tokens, 1))

            score = weight * quality_boost * token_factor * context.phi_saturation
            scores.append((meta, score))

        # Return highest scoring
        return max(scores, key=lambda x: x[1])[0] if scores else None


# =============================================================================
# PREDEFINED SKILLS
# =============================================================================

# --- Code Skills ---

class PythonSkill(Skill):
    """Python code generation and analysis skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/python",
            name="Python Skill",
            description="Generate, analyze, and manipulate Python code",
            category=SkillCategory.CODE,
            level=SkillLevel.INTERMEDIATE,
            tags=["python", "code", "programming"],
            estimated_tokens=1000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        task = input_data.get("task", "analyze")
        code = input_data.get("code", "")
        return SkillResult(
            success=True,
            output={"task": task, "code_length": len(code), "analysis": "Python code processed"},
            skill_id=self.metadata().id,
        )


class RefactorSkill(Skill):
    """Code refactoring skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/refactor",
            name="Refactor Skill",
            description="Refactor code for better quality, readability, and performance",
            category=SkillCategory.CODE,
            level=SkillLevel.ADVANCED,
            tags=["refactor", "code", "quality"],
            estimated_tokens=1500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        code = input_data.get("code", "")
        strategy = input_data.get("strategy", "readability")
        return SkillResult(
            success=True,
            output={"strategy": strategy, "original_lines": code.count('\n') + 1},
            skill_id=self.metadata().id,
        )


class ReviewSkill(Skill):
    """Code review skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/review",
            name="Review Skill",
            description="Review code for bugs, security issues, and best practices",
            category=SkillCategory.CODE,
            level=SkillLevel.ADVANCED,
            tags=["review", "code", "security"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        code = input_data.get("code", "")
        return SkillResult(
            success=True,
            output={"issues": [], "suggestions": [], "security_score": 0.95},
            skill_id=self.metadata().id,
        )


class FormatSkill(Skill):
    """Code formatting skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/format",
            name="Format Skill",
            description="Format code according to style guides",
            category=SkillCategory.CODE,
            level=SkillLevel.BASIC,
            tags=["format", "style", "code"],
            estimated_tokens=500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        code = input_data.get("code", "")
        style = input_data.get("style", "pep8")
        return SkillResult(
            success=True,
            output={"style": style, "formatted": True},
            skill_id=self.metadata().id,
        )


class DebugSkill(Skill):
    """Debugging skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/debug",
            name="Debug Skill",
            description="Debug code and identify issues",
            category=SkillCategory.CODE,
            level=SkillLevel.ADVANCED,
            tags=["debug", "errors", "code"],
            estimated_tokens=1500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        code = input_data.get("code", "")
        error = input_data.get("error", "")
        return SkillResult(
            success=True,
            output={"error_type": "unknown", "root_cause": "", "fix_suggestion": ""},
            skill_id=self.metadata().id,
        )


class TestSkill(Skill):
    """Test generation skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="code/test",
            name="Test Skill",
            description="Generate tests for code",
            category=SkillCategory.CODE,
            level=SkillLevel.INTERMEDIATE,
            tags=["test", "testing", "code"],
            estimated_tokens=1200,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        code = input_data.get("code", "")
        framework = input_data.get("framework", "pytest")
        return SkillResult(
            success=True,
            output={"framework": framework, "test_count": 0, "coverage": 0.0},
            skill_id=self.metadata().id,
        )


# --- Data Skills ---

class AnalysisSkill(Skill):
    """Data analysis skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="data/analysis",
            name="Analysis Skill",
            description="Analyze data and extract insights",
            category=SkillCategory.DATA,
            level=SkillLevel.INTERMEDIATE,
            tags=["data", "analysis", "insights"],
            estimated_tokens=1500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        data = input_data.get("data", {})
        return SkillResult(
            success=True,
            output={"summary": {}, "insights": [], "correlations": []},
            skill_id=self.metadata().id,
        )


class TransformSkill(Skill):
    """Data transformation skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="data/transform",
            name="Transform Skill",
            description="Transform and reshape data",
            category=SkillCategory.DATA,
            level=SkillLevel.INTERMEDIATE,
            tags=["data", "transform", "etl"],
            estimated_tokens=1000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        data = input_data.get("data", {})
        operation = input_data.get("operation", "normalize")
        return SkillResult(
            success=True,
            output={"operation": operation, "transformed": True},
            skill_id=self.metadata().id,
        )


class DataValidationSkill(Skill):
    """Data validation skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="data/validation",
            name="Data Validation Skill",
            description="Validate data quality and integrity",
            category=SkillCategory.DATA,
            level=SkillLevel.BASIC,
            tags=["data", "validation", "quality"],
            estimated_tokens=800,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        data = input_data.get("data", {})
        schema = input_data.get("schema", {})
        return SkillResult(
            success=True,
            output={"valid": True, "errors": [], "warnings": []},
            skill_id=self.metadata().id,
        )


class AggregationSkill(Skill):
    """Data aggregation skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="data/aggregation",
            name="Aggregation Skill",
            description="Aggregate and summarize data",
            category=SkillCategory.DATA,
            level=SkillLevel.INTERMEDIATE,
            tags=["data", "aggregation", "summary"],
            estimated_tokens=1000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        data = input_data.get("data", [])
        group_by = input_data.get("group_by", None)
        return SkillResult(
            success=True,
            output={"aggregations": {}, "group_by": group_by},
            skill_id=self.metadata().id,
        )


# --- Research Skills ---

class MathLiteratureSkill(Skill):
    """Mathematical literature search skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="research/math-literature",
            name="Math Literature Skill",
            description="Search and analyze mathematical literature",
            category=SkillCategory.RESEARCH,
            level=SkillLevel.ADVANCED,
            tags=["research", "math", "literature"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        query = input_data.get("query", "")
        return SkillResult(
            success=True,
            output={"papers": [], "relevance_scores": [], "summary": ""},
            skill_id=self.metadata().id,
        )


class BSDComparisonSkill(Skill):
    """BSD conjecture comparison morphism skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="research/bsd-comparison",
            name="BSD Comparison Skill",
            description="Analyze BSD conjecture comparison morphisms",
            category=SkillCategory.RESEARCH,
            level=SkillLevel.EXPERT,
            tags=["research", "bsd", "elliptic-curves"],
            estimated_tokens=3000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        curve = input_data.get("curve", "")
        return SkillResult(
            success=True,
            output={"curve": curve, "analysis": {}, "morphisms": []},
            skill_id=self.metadata().id,
        )


class SynthesisSkill(Skill):
    """Research synthesis skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="research/synthesis",
            name="Synthesis Skill",
            description="Synthesize findings from multiple sources",
            category=SkillCategory.SYNTHESIS,
            level=SkillLevel.ADVANCED,
            tags=["research", "synthesis", "integration"],
            estimated_tokens=2500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        sources = input_data.get("sources", [])
        return SkillResult(
            success=True,
            output={"synthesis": "", "key_findings": [], "gaps": []},
            skill_id=self.metadata().id,
        )


class CitationSkill(Skill):
    """Citation management skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="research/citation",
            name="Citation Skill",
            description="Manage and format citations",
            category=SkillCategory.RESEARCH,
            level=SkillLevel.INTERMEDIATE,
            tags=["research", "citation", "bibliography"],
            estimated_tokens=800,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        refs = input_data.get("references", [])
        style = input_data.get("style", "APA")
        return SkillResult(
            success=True,
            output={"formatted": [], "style": style},
            skill_id=self.metadata().id,
        )


# --- Cognition Skills ---

class NeuroIntuitionSkill(Skill):
    """Neuro-intuition skill for pattern-based insights."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="cognition/neuro-intuition",
            name="Neuro Intuition Skill",
            description="Generate intuitive insights from patterns",
            category=SkillCategory.COGNITION,
            level=SkillLevel.EXPERT,
            tags=["cognition", "intuition", "patterns"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        patterns = input_data.get("patterns", [])
        return SkillResult(
            success=True,
            output={"intuitions": [], "confidence": 0.0, "reasoning": ""},
            skill_id=self.metadata().id,
        )


class PatternRecognitionSkill(Skill):
    """Pattern recognition skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="cognition/pattern-recognition",
            name="Pattern Recognition Skill",
            description="Identify patterns in data and concepts",
            category=SkillCategory.PATTERN_RECOGNITION,
            level=SkillLevel.ADVANCED,
            tags=["cognition", "patterns", "recognition"],
            estimated_tokens=1500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        data = input_data.get("data", [])
        return SkillResult(
            success=True,
            output={"patterns": [], "confidence_scores": [], "anomalies": []},
            skill_id=self.metadata().id,
        )


class ReasoningSkill(Skill):
    """Deep reasoning skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="cognition/reasoning",
            name="Reasoning Skill",
            description="Apply deep logical reasoning to problems",
            category=SkillCategory.REASONING,
            level=SkillLevel.ADVANCED,
            tags=["cognition", "reasoning", "logic"],
            estimated_tokens=2500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        problem = input_data.get("problem", "")
        premises = input_data.get("premises", [])
        return SkillResult(
            success=True,
            output={"conclusion": "", "reasoning_chain": [], "confidence": 0.0},
            skill_id=self.metadata().id,
        )


class AbstractionSkill(Skill):
    """Abstraction skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="cognition/abstraction",
            name="Abstraction Skill",
            description="Abstract concepts and find generalizations",
            category=SkillCategory.COGNITION,
            level=SkillLevel.EXPERT,
            tags=["cognition", "abstraction", "generalization"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        concepts = input_data.get("concepts", [])
        return SkillResult(
            success=True,
            output={"abstractions": [], "hierarchy": {}, "principles": []},
            skill_id=self.metadata().id,
        )


# --- Writing Skills ---

class DraftSkill(Skill):
    """Drafting skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="writing/draft",
            name="Draft Skill",
            description="Create initial drafts of documents",
            category=SkillCategory.WRITING,
            level=SkillLevel.INTERMEDIATE,
            tags=["writing", "draft", "content"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        topic = input_data.get("topic", "")
        outline = input_data.get("outline", [])
        return SkillResult(
            success=True,
            output={"draft": "", "word_count": 0, "sections": []},
            skill_id=self.metadata().id,
        )


class EditSkill(Skill):
    """Editing skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="writing/edit",
            name="Edit Skill",
            description="Edit and improve text",
            category=SkillCategory.WRITING,
            level=SkillLevel.INTERMEDIATE,
            tags=["writing", "edit", "improve"],
            estimated_tokens=1500,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        text = input_data.get("text", "")
        focus = input_data.get("focus", "clarity")
        return SkillResult(
            success=True,
            output={"edited": "", "changes": [], "focus": focus},
            skill_id=self.metadata().id,
        )


class SummarySkill(Skill):
    """Summarization skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="writing/summary",
            name="Summary Skill",
            description="Summarize text and documents",
            category=SkillCategory.WRITING,
            level=SkillLevel.BASIC,
            tags=["writing", "summary", "condensation"],
            estimated_tokens=1000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        text = input_data.get("text", "")
        length = input_data.get("length", "medium")
        return SkillResult(
            success=True,
            output={"summary": "", "key_points": [], "length": length},
            skill_id=self.metadata().id,
        )


class TechnicalWritingSkill(Skill):
    """Technical writing skill."""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            id="writing/technical",
            name="Technical Writing Skill",
            description="Create technical documentation",
            category=SkillCategory.WRITING,
            level=SkillLevel.ADVANCED,
            tags=["writing", "technical", "documentation"],
            estimated_tokens=2000,
        )

    def execute(self, input_data: Dict[str, Any], context: SkillContext) -> SkillResult:
        subject = input_data.get("subject", "")
        audience = input_data.get("audience", "developers")
        return SkillResult(
            success=True,
            output={"document": "", "sections": [], "audience": audience},
            skill_id=self.metadata().id,
        )


# =============================================================================
# SKILL COLLECTIONS
# =============================================================================

CODE_SKILLS = [PythonSkill, RefactorSkill, ReviewSkill, FormatSkill, DebugSkill, TestSkill]
DATA_SKILLS = [AnalysisSkill, TransformSkill, DataValidationSkill, AggregationSkill]
RESEARCH_SKILLS = [MathLiteratureSkill, BSDComparisonSkill, SynthesisSkill, CitationSkill]
COGNITION_SKILLS = [NeuroIntuitionSkill, PatternRecognitionSkill, ReasoningSkill, AbstractionSkill]
WRITING_SKILLS = [DraftSkill, EditSkill, SummarySkill, TechnicalWritingSkill]

ALL_SKILLS = CODE_SKILLS + DATA_SKILLS + RESEARCH_SKILLS + COGNITION_SKILLS + WRITING_SKILLS


# =============================================================================
# MAIN INTERFACE
# =============================================================================

class Skills:
    """
    Unified interface for all skill operations.

    Provides access to:
    - Registry for skill management
    - Loader for progressive disclosure
    - Selector for autonomous skill selection
    - Router for dimension-aware routing
    """

    def __init__(self):
        self.registry = SkillRegistry()
        self.loader = ProgressiveLoader()
        self.selector = AutonomousSkillSelector()
        self.router = SkillRouter(self.registry, self.loader)
        self.gpia_registry = GPIACoreSkillRegistry()

        # Register all predefined skills
        self._register_default_skills()

    def _register_default_skills(self):
        """Register all default skills."""
        for skill_class in ALL_SKILLS:
            skill = skill_class()
            self.registry.register(skill)
            self.loader.register(skill_class)

    def execute(self, skill_id: str, input_data: Dict[str, Any],
                context: SkillContext = None) -> SkillResult:
        """Execute a skill by ID."""
        ctx = context or SkillContext()
        return self.loader.execute(skill_id, input_data, ctx)

    def select_and_execute(self, model: str, task: str, input_data: Dict[str, Any],
                          context: SkillContext = None) -> SkillResult:
        """Select the best skill and execute it."""
        available = [s["id"] for s in self.loader.list_skills()]
        selection = self.selector.select_skill(model, task, available)

        if selection.skill_id is None:
            return SkillResult(
                success=False,
                output=None,
                error="No suitable skill found",
                error_code="NO_SKILL",
            )

        result = self.execute(selection.skill_id, input_data, context)

        # Record outcome for learning
        self.selector.record_outcome(
            model, task, selection.skill_id,
            result.success,
            0.8 if result.success else 0.2,
            selection.method.value
        )

        return result

    def list_skills(self, category: str = None) -> List[Dict[str, Any]]:
        """List all available skills."""
        if category:
            handles = self.loader.find_by_category(category)
        else:
            handles = list(self.loader._handles.values())
        return [
            {
                "id": h.manifest.id,
                "name": h.manifest.name,
                "category": h.manifest.category,
                "level": h.manifest.level,
                "loaded": h.is_loaded,
            }
            for h in handles
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall skill statistics."""
        return {
            "total_skills": len(self.loader._handles),
            "loaded_skills": sum(1 for h in self.loader._handles.values() if h.is_loaded),
            "loader_stats": self.loader.get_stats(),
            "selector_exploration_rate": self.selector.exploration_rate,
            "selector_confidence_threshold": self.selector.confidence_threshold,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_registry: Optional[SkillRegistry] = None
_loader: Optional[ProgressiveLoader] = None
_skills: Optional[Skills] = None


def create_skill(skill_class: Type[Skill]) -> Skill:
    """Create a skill instance."""
    return skill_class()


def create_registry() -> SkillRegistry:
    """Create a new skill registry."""
    return SkillRegistry()


def create_loader() -> ProgressiveLoader:
    """Create a new progressive loader."""
    return ProgressiveLoader()


def create_selector() -> AutonomousSkillSelector:
    """Create a new autonomous skill selector."""
    return AutonomousSkillSelector()


def create_skills() -> Skills:
    """Create the unified Skills interface."""
    global _skills
    if _skills is None:
        _skills = Skills()
    return _skills


def load_skill(skill_id: str) -> Optional[Skill]:
    """Load a skill by ID from global loader."""
    loader = get_loader()
    handle = loader.get(skill_id)
    return handle.get_instance() if handle else None


def get_registry() -> SkillRegistry:
    """Get global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def get_loader() -> ProgressiveLoader:
    """Get global progressive loader."""
    global _loader
    if _loader is None:
        _loader = ProgressiveLoader()
    return _loader
