"""
ASIOS Government System - Minister Cabinet

Role-based model routing with semantic similarity matching.
Implements a cabinet of specialized ministers for task delegation.

Architecture:
    President (Arbiter) → Routes to appropriate Minister
    Ministers → Specialized processing based on semantic domain
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path

import numpy as np

# Import calculator constants
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sovereign_pio.constants import PHI, BETA


class MetabolicCost(Enum):
    """Computational cost classification."""
    MICRO = "micro"      # < 100 tokens, instant
    LOW = "low"          # 100-500 tokens
    MEDIUM = "medium"    # 500-2000 tokens
    HIGH = "high"        # 2000-8000 tokens
    EXTREME = "extreme"  # > 8000 tokens


@dataclass
class Minister:
    """A specialized minister in the cabinet."""
    id: str
    title: str
    domain: str
    description: str
    embedding: Optional[np.ndarray] = None
    model_preference: Optional[str] = None
    metabolic_budget: MetabolicCost = MetabolicCost.MEDIUM

    # Performance tracking
    tasks_handled: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "domain": self.domain,
            "description": self.description,
            "model_preference": self.model_preference,
            "metabolic_budget": self.metabolic_budget.value,
            "tasks_handled": self.tasks_handled,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class CabinetDecision:
    """Result of cabinet routing decision."""
    minister: Minister
    confidence: float
    reasoning: str
    alternatives: List[Tuple[Minister, float]] = field(default_factory=list)
    metabolic_estimate: MetabolicCost = MetabolicCost.MEDIUM


# Default cabinet configuration
DEFAULT_CABINET = [
    Minister(
        id="president",
        title="President",
        domain="arbitration",
        description="Final arbiter for conflicts and high-stakes decisions. Routes ambiguous tasks.",
        metabolic_budget=MetabolicCost.HIGH,
    ),
    Minister(
        id="prime_minister",
        title="Prime Minister of Reason",
        domain="logic",
        description="Micro-cost logical operations. Quick reasoning chains.",
        metabolic_budget=MetabolicCost.MICRO,
    ),
    Minister(
        id="strategist",
        title="Chief Strategist",
        domain="planning",
        description="Complex multi-step planning and strategy formulation.",
        metabolic_budget=MetabolicCost.HIGH,
    ),
    Minister(
        id="mathematician",
        title="Minister of Mathematics",
        domain="computation",
        description="Formal proofs, calculations, and mathematical verification.",
        metabolic_budget=MetabolicCost.MEDIUM,
    ),
    Minister(
        id="truth",
        title="Minister of Truth",
        domain="verification",
        description="Fact-checking, source verification, and truth assessment.",
        metabolic_budget=MetabolicCost.MEDIUM,
    ),
    Minister(
        id="intelligence",
        title="Minister of Intelligence",
        domain="analysis",
        description="Deep analysis, simulation, and red-teaming.",
        metabolic_budget=MetabolicCost.HIGH,
    ),
    Minister(
        id="constitution",
        title="Minister of Constitution",
        domain="compliance",
        description="Legal and policy compliance checks.",
        metabolic_budget=MetabolicCost.LOW,
    ),
    Minister(
        id="creativity",
        title="Minister of Creativity",
        domain="generation",
        description="Creative writing, ideation, and novel synthesis.",
        metabolic_budget=MetabolicCost.HIGH,
    ),
    Minister(
        id="memory",
        title="Minister of Memory",
        domain="retrieval",
        description="Memory retrieval and context management.",
        metabolic_budget=MetabolicCost.LOW,
    ),
    Minister(
        id="communication",
        title="Minister of Communication",
        domain="translation",
        description="Translation, summarization, and format conversion.",
        metabolic_budget=MetabolicCost.MEDIUM,
    ),
]


def _simple_embed(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic embedding from text hash."""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], 'big')
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class Government:
    """
    ASIOS Government - Minister Cabinet System.

    Routes tasks to specialized ministers based on semantic similarity.
    Uses PHI-based confidence thresholds for fallback routing.

    Example:
        gov = Government()
        decision = gov.route("Calculate the factorial of 10")
        print(f"Routed to: {decision.minister.title}")
    """

    # Confidence thresholds (PHI-based)
    HIGH_CONFIDENCE = 0.7        # Clear match
    MEDIUM_CONFIDENCE = PHI - 1  # 0.618 - borderline
    LOW_CONFIDENCE = BETA        # 0.236 - fallback to president

    def __init__(
        self,
        cabinet: Optional[List[Minister]] = None,
        embedding_dim: int = 384,
    ):
        self.cabinet = cabinet or [Minister(**m.__dict__) for m in DEFAULT_CABINET]
        self.embedding_dim = embedding_dim
        self.president = self._find_president()

        # Initialize embeddings
        self._initialize_embeddings()

        # Statistics
        self.total_decisions = 0
        self.president_interventions = 0

    def _find_president(self) -> Minister:
        """Find the president in the cabinet."""
        for m in self.cabinet:
            if m.id == "president":
                return m
        # Create default president if not found
        president = Minister(
            id="president",
            title="President",
            domain="arbitration",
            description="Final arbiter",
        )
        self.cabinet.insert(0, president)
        return president

    def _initialize_embeddings(self):
        """Initialize embeddings for all ministers."""
        for minister in self.cabinet:
            if minister.embedding is None:
                # Create embedding from domain + description
                text = f"{minister.domain} {minister.description}"
                minister.embedding = _simple_embed(text, self.embedding_dim)

    def route(self, task: str, context: Optional[str] = None) -> CabinetDecision:
        """
        Route a task to the appropriate minister.

        Args:
            task: The task description
            context: Optional additional context

        Returns:
            CabinetDecision with selected minister and confidence
        """
        self.total_decisions += 1

        # Embed the task
        full_text = f"{task} {context}" if context else task
        task_embedding = _simple_embed(full_text, self.embedding_dim)

        # Compute similarities to all ministers (except president)
        scores: List[Tuple[Minister, float]] = []
        for minister in self.cabinet:
            if minister.id == "president":
                continue

            sim = cosine_similarity(task_embedding, minister.embedding)
            scores.append((minister, sim))

        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Decision logic
        best_minister, best_score = scores[0]

        if best_score >= self.HIGH_CONFIDENCE:
            # Clear match
            return CabinetDecision(
                minister=best_minister,
                confidence=best_score,
                reasoning=f"High confidence match to {best_minister.domain}",
                alternatives=scores[1:3],
                metabolic_estimate=best_minister.metabolic_budget,
            )

        elif best_score >= self.MEDIUM_CONFIDENCE:
            # Check if there's a close second
            if len(scores) > 1:
                second_minister, second_score = scores[1]
                if best_score - second_score < 0.1:
                    # Too close, escalate to president
                    self.president_interventions += 1
                    return CabinetDecision(
                        minister=self.president,
                        confidence=best_score,
                        reasoning=f"Ambiguous between {best_minister.title} and {second_minister.title}",
                        alternatives=scores[:3],
                        metabolic_estimate=self.president.metabolic_budget,
                    )

            # Moderate confidence, proceed with best match
            return CabinetDecision(
                minister=best_minister,
                confidence=best_score,
                reasoning=f"Moderate confidence match to {best_minister.domain}",
                alternatives=scores[1:3],
                metabolic_estimate=best_minister.metabolic_budget,
            )

        else:
            # Low confidence, route to president
            self.president_interventions += 1
            return CabinetDecision(
                minister=self.president,
                confidence=best_score,
                reasoning=f"Low confidence ({best_score:.3f}), escalating to President",
                alternatives=scores[:3],
                metabolic_estimate=self.president.metabolic_budget,
            )

    def record_outcome(self, minister_id: str, success: bool, confidence: float):
        """Record the outcome of a task for learning."""
        for minister in self.cabinet:
            if minister.id == minister_id:
                minister.tasks_handled += 1
                # Running average of confidence
                n = minister.tasks_handled
                minister.avg_confidence = (
                    (minister.avg_confidence * (n - 1) + confidence) / n
                )
                break

    def get_minister(self, minister_id: str) -> Optional[Minister]:
        """Get a minister by ID."""
        for minister in self.cabinet:
            if minister.id == minister_id:
                return minister
        return None

    def list_ministers(self) -> List[Dict[str, Any]]:
        """List all ministers with their stats."""
        return [m.to_dict() for m in self.cabinet]

    def stats(self) -> Dict[str, Any]:
        """Get government statistics."""
        return {
            "total_decisions": self.total_decisions,
            "president_interventions": self.president_interventions,
            "intervention_rate": (
                self.president_interventions / max(1, self.total_decisions)
            ),
            "cabinet_size": len(self.cabinet),
            "confidence_thresholds": {
                "high": self.HIGH_CONFIDENCE,
                "medium": self.MEDIUM_CONFIDENCE,
                "low": self.LOW_CONFIDENCE,
            },
        }
