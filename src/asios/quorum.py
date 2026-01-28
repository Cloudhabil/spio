"""
Quorum Voting System

Consensus voting for multi-model decisions.
Enables high-confidence decisions through model agreement.

Features:
- Multiple model voting
- Confidence levels (HIGH, MEDIUM, CONFLICTED)
- Minority report preservation
- Configurable consensus rules
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels based on model agreement."""
    HIGH = "high"           # All models agree
    MEDIUM = "medium"       # Majority agrees (≥2/3)
    LOW = "low"             # Simple majority
    CONFLICTED = "conflicted"  # No majority


@dataclass
class Vote:
    """A single vote from a model."""
    model_id: str
    answer: str
    confidence: float  # Model's self-reported confidence
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuorumResult:
    """Result of a quorum vote."""
    decision: str
    confidence_level: ConfidenceLevel
    vote_count: int
    agreement_ratio: float
    votes: List[Vote] = field(default_factory=list)
    minority_reports: List[Vote] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence_level": self.confidence_level.value,
            "vote_count": self.vote_count,
            "agreement_ratio": self.agreement_ratio,
            "votes": [
                {"model": v.model_id, "answer": v.answer, "confidence": v.confidence}
                for v in self.votes
            ],
            "minority_count": len(self.minority_reports),
        }


# Voter function type
Voter = Callable[[str], Awaitable[Vote]]


class Quorum:
    """
    Quorum Voting System.

    Aggregates votes from multiple models to reach consensus decisions
    with quantified confidence.

    Example:
        quorum = Quorum()

        # Register voters (models)
        quorum.register_voter("gpt4", gpt4_vote_fn)
        quorum.register_voter("claude", claude_vote_fn)
        quorum.register_voter("llama", llama_vote_fn)

        # Vote on a question
        result = await quorum.vote("Is this code secure?")
        print(f"Decision: {result.decision}")
        print(f"Confidence: {result.confidence_level.value}")
    """

    def __init__(
        self,
        min_votes: int = 2,
        high_threshold: float = 1.0,    # 100% agreement for HIGH
        medium_threshold: float = 0.67,  # 67% for MEDIUM
    ):
        self.min_votes = min_votes
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        self.voters: Dict[str, Voter] = {}

        # Statistics
        self.total_votes = 0
        self.high_confidence_count = 0
        self.conflicted_count = 0

    def register_voter(self, model_id: str, voter: Voter):
        """Register a voting model."""
        self.voters[model_id] = voter

    def unregister_voter(self, model_id: str):
        """Remove a voting model."""
        self.voters.pop(model_id, None)

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Simple normalization: lowercase, strip, remove punctuation
        normalized = answer.lower().strip()
        # Remove common variations
        for prefix in ["yes,", "no,", "i think", "the answer is"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        return normalized

    def _compute_consensus(self, votes: List[Vote]) -> QuorumResult:
        """Compute consensus from votes."""
        if not votes:
            return QuorumResult(
                decision="",
                confidence_level=ConfidenceLevel.CONFLICTED,
                vote_count=0,
                agreement_ratio=0.0,
            )

        # Group by normalized answer
        groups: Dict[str, List[Vote]] = {}
        for vote in votes:
            key = self._normalize_answer(vote.answer)
            if key not in groups:
                groups[key] = []
            groups[key].append(vote)

        # Find majority
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        majority_answer, majority_votes = sorted_groups[0]

        # Calculate agreement ratio
        agreement_ratio = len(majority_votes) / len(votes)

        # Determine confidence level
        if agreement_ratio >= self.high_threshold:
            confidence_level = ConfidenceLevel.HIGH
            self.high_confidence_count += 1
        elif agreement_ratio >= self.medium_threshold:
            confidence_level = ConfidenceLevel.MEDIUM
        elif agreement_ratio > 0.5:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.CONFLICTED
            self.conflicted_count += 1

        # Collect minority reports
        minority_reports = []
        for answer, group in sorted_groups[1:]:
            minority_reports.extend(group)

        # Use the highest confidence vote's exact answer
        best_vote = max(majority_votes, key=lambda v: v.confidence)

        return QuorumResult(
            decision=best_vote.answer,
            confidence_level=confidence_level,
            vote_count=len(votes),
            agreement_ratio=agreement_ratio,
            votes=majority_votes,
            minority_reports=minority_reports,
        )

    async def vote(
        self,
        question: str,
        context: Optional[str] = None,
        required_voters: Optional[List[str]] = None,
    ) -> QuorumResult:
        """
        Conduct a vote on a question.

        Args:
            question: The question to vote on
            context: Optional context for the question
            required_voters: Specific voters to use (default: all)

        Returns:
            QuorumResult with decision and confidence
        """
        self.total_votes += 1

        # Determine which voters to use
        voter_ids = required_voters or list(self.voters.keys())

        # Collect votes
        votes: List[Vote] = []
        for voter_id in voter_ids:
            if voter_id not in self.voters:
                continue

            try:
                prompt = question
                if context:
                    prompt = f"Context: {context}\n\nQuestion: {question}"

                vote = await self.voters[voter_id](prompt)
                vote.model_id = voter_id
                votes.append(vote)
            except Exception as e:
                # Record failed vote attempt
                votes.append(Vote(
                    model_id=voter_id,
                    answer="[VOTE_FAILED]",
                    confidence=0.0,
                    reasoning=str(e),
                ))

        # Filter out failed votes
        valid_votes = [v for v in votes if v.answer != "[VOTE_FAILED]"]

        # Check minimum votes
        if len(valid_votes) < self.min_votes:
            return QuorumResult(
                decision="",
                confidence_level=ConfidenceLevel.CONFLICTED,
                vote_count=len(valid_votes),
                agreement_ratio=0.0,
                votes=valid_votes,
                minority_reports=[],
            )

        return self._compute_consensus(valid_votes)

    def vote_sync(self, question: str, votes: List[Vote]) -> QuorumResult:
        """
        Synchronous voting with pre-collected votes.

        Useful when votes are already available.
        """
        self.total_votes += 1
        return self._compute_consensus(votes)

    def stats(self) -> Dict[str, Any]:
        """Get quorum statistics."""
        return {
            "registered_voters": len(self.voters),
            "total_votes": self.total_votes,
            "high_confidence_count": self.high_confidence_count,
            "conflicted_count": self.conflicted_count,
            "high_confidence_rate": (
                self.high_confidence_count / max(1, self.total_votes)
            ),
        }


def simple_voter(model_id: str, response_fn: Callable[[str], str]) -> Voter:
    """
    Create a simple voter from a response function.

    Args:
        model_id: Model identifier
        response_fn: Function that takes prompt and returns answer

    Returns:
        Voter function
    """
    async def voter(prompt: str) -> Vote:
        answer = response_fn(prompt)
        return Vote(
            model_id=model_id,
            answer=answer,
            confidence=0.8,  # Default confidence
        )

    return voter
