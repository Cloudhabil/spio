#!/usr/bin/env python3
"""
Ephemeral Onion Subnet - Mixture of Experts with Learned Routing

Implements sparse MoE architecture for intent classification with:
- Gating Network (learned router)
- Expert Subnets (specialized per intent)
- Sparse Activation (top-k routing)
- Load Balancing Loss (prevents expert collapse)

Based on:
- Shazeer et al. (2017) "Outrageously Large Neural Networks"
- Hinton et al. (2018) "Matrix Capsules with EM Routing"

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import json

# Import Brahim constants
from .wavelength_integration import (
    BRAHIM_SEQUENCE,
    SUM_CONSTANT,
    CENTER,
    PHI,
    DIMENSION,
)


# =============================================================================
# INTENT ENUMERATION
# =============================================================================

class Intent(Enum):
    """Supported intents for classification."""
    PHYSICS = "physics"
    COSMOLOGY = "cosmology"
    YANG_MILLS = "yang_mills"
    MIRROR = "mirror"
    SEQUENCE = "sequence"
    VERIFY = "verify"
    HELP = "help"
    UNKNOWN = "unknown"


# =============================================================================
# EXPERT SUBNET
# =============================================================================

@dataclass
class ExpertConfig:
    """Configuration for an expert subnet."""
    intent: Intent
    hidden_dim: int = 32
    dropout: float = 0.1
    keywords: List[str] = field(default_factory=list)
    weight_decay: float = 0.01


class ExpertSubnet:
    """
    Specialized expert network for a single intent.

    Architecture:
        Input (10-dim) → Dense(32) → ReLU → Dense(1) → Sigmoid

    Each expert learns to recognize patterns specific to its intent.
    """

    def __init__(self, config: ExpertConfig):
        self.config = config
        self.intent = config.intent

        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(DIMENSION, config.hidden_dim) * np.sqrt(2.0 / DIMENSION)
        self.b1 = np.zeros(config.hidden_dim)
        self.W2 = np.random.randn(config.hidden_dim, 1) * np.sqrt(2.0 / config.hidden_dim)
        self.b2 = np.zeros(1)

        # Keyword embeddings for this expert
        self.keyword_embeddings = self._init_keyword_embeddings(config.keywords)

        # Training statistics
        self.usage_count = 0
        self.total_loss = 0.0

    def _init_keyword_embeddings(self, keywords: List[str]) -> Dict[str, np.ndarray]:
        """Initialize keyword embeddings in Brahim space."""
        embeddings = {}
        for i, kw in enumerate(keywords):
            # Map keyword to position in Brahim sequence
            idx = i % len(BRAHIM_SEQUENCE)
            base = BRAHIM_SEQUENCE[idx] / SUM_CONSTANT

            # Create embedding with keyword-specific perturbation
            emb = np.zeros(DIMENSION)
            emb[idx] = base
            emb[(idx + 1) % DIMENSION] = base * PHI
            emb[(idx + 2) % DIMENSION] = base / PHI

            embeddings[kw.lower()] = emb / (np.linalg.norm(emb) + 1e-8)

        return embeddings

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass through expert.

        Args:
            x: Input embedding (10-dim)

        Returns:
            Confidence score [0, 1]
        """
        # Layer 1: Dense + ReLU
        h = np.maximum(0, x @ self.W1 + self.b1)

        # Dropout during forward (simplified)
        if self.config.dropout > 0:
            mask = np.random.binomial(1, 1 - self.config.dropout, h.shape)
            h = h * mask / (1 - self.config.dropout)

        # Layer 2: Dense + Sigmoid
        logit = h @ self.W2 + self.b2
        confidence = 1 / (1 + np.exp(-logit[0]))

        self.usage_count += 1
        return confidence

    def compute_keyword_similarity(self, text: str) -> float:
        """Compute similarity between text and expert's keywords."""
        text_lower = text.lower()
        similarities = []

        for kw, emb in self.keyword_embeddings.items():
            if kw in text_lower:
                similarities.append(1.0)
            else:
                # Partial match score
                for word in text_lower.split():
                    if kw in word or word in kw:
                        similarities.append(0.5)
                        break

        return max(similarities) if similarities else 0.0

    def update(self, x: np.ndarray, target: float, lr: float = 0.01):
        """
        Update expert weights via gradient descent.

        Args:
            x: Input embedding
            target: Target label (1 if this expert's intent, 0 otherwise)
            lr: Learning rate
        """
        # Forward pass (without dropout)
        h = np.maximum(0, x @ self.W1 + self.b1)
        logit = h @ self.W2 + self.b2
        pred = 1 / (1 + np.exp(-logit[0]))

        # Binary cross-entropy loss
        eps = 1e-8
        loss = -target * np.log(pred + eps) - (1 - target) * np.log(1 - pred + eps)
        self.total_loss += loss

        # Backpropagation
        d_logit = pred - target

        # Gradients for W2, b2
        d_W2 = h.reshape(-1, 1) * d_logit
        d_b2 = np.array([d_logit])

        # Gradients for W1, b1
        d_h = d_logit * self.W2.flatten()
        d_h_pre = d_h * (h > 0).astype(float)  # ReLU gradient
        d_W1 = np.outer(x, d_h_pre)
        d_b1 = d_h_pre

        # Weight decay (L2 regularization)
        d_W1 += self.config.weight_decay * self.W1
        d_W2 += self.config.weight_decay * self.W2

        # Update weights
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2


# =============================================================================
# GATING NETWORK (ROUTER)
# =============================================================================

class GatingNetwork:
    """
    Learned router that determines which experts to activate.

    Architecture:
        Input (10-dim) → Dense(num_experts) → Softmax → Top-K selection

    Implements:
        - Soft routing with temperature
        - Noise injection for exploration
        - Load balancing auxiliary loss
    """

    def __init__(self, input_dim: int = DIMENSION, num_experts: int = 7,
                 temperature: float = 1.0, noise_std: float = 0.1):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.noise_std = noise_std

        # Gate weights with Xavier init
        self.W_gate = np.random.randn(input_dim, num_experts) * np.sqrt(2.0 / input_dim)
        self.b_gate = np.zeros(num_experts)

        # Expert usage tracking for load balancing
        self.expert_usage = np.zeros(num_experts)
        self.total_samples = 0

    def forward(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Compute routing probabilities.

        Args:
            x: Input embedding (10-dim)
            add_noise: Whether to add exploration noise

        Returns:
            Probability distribution over experts
        """
        logits = x @ self.W_gate + self.b_gate

        # Add noise for exploration during training
        if add_noise and self.noise_std > 0:
            noise = np.random.randn(self.num_experts) * self.noise_std
            logits = logits + noise

        # Temperature-scaled softmax
        logits = logits / self.temperature
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def select_top_k(self, probs: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-k experts based on routing probabilities.

        Args:
            probs: Probability distribution over experts
            k: Number of experts to select

        Returns:
            Tuple of (selected probabilities, selected indices)
        """
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_probs = probs[top_k_indices]

        # Renormalize selected probabilities
        top_k_probs = top_k_probs / (np.sum(top_k_probs) + 1e-8)

        # Update usage statistics
        for idx in top_k_indices:
            self.expert_usage[idx] += 1
        self.total_samples += 1

        return top_k_probs, top_k_indices

    def compute_load_balance_loss(self) -> float:
        """
        Compute auxiliary loss to encourage balanced expert usage.

        Returns:
            Load balance loss (coefficient of variation squared)
        """
        if self.total_samples == 0:
            return 0.0

        usage_fraction = self.expert_usage / self.total_samples
        mean_usage = np.mean(usage_fraction)

        if mean_usage == 0:
            return 0.0

        # Coefficient of variation
        cv = np.std(usage_fraction) / mean_usage
        return cv ** 2

    def update(self, x: np.ndarray, selected_indices: np.ndarray,
               rewards: np.ndarray, lr: float = 0.01):
        """
        Update gate weights using REINFORCE-style gradient.

        Args:
            x: Input embedding
            selected_indices: Which experts were selected
            rewards: Reward signal for each selected expert
            lr: Learning rate
        """
        probs = self.forward(x, add_noise=False)

        # Policy gradient
        for i, idx in enumerate(selected_indices):
            # Increase probability of experts that gave high reward
            grad = rewards[i] * (1 - probs[idx]) * x
            self.W_gate[:, idx] += lr * grad
            self.b_gate[idx] += lr * rewards[i] * (1 - probs[idx])


# =============================================================================
# EPHEMERAL ONION SUBNET
# =============================================================================

# Expert configurations with domain-specific keywords
EXPERT_CONFIGS = {
    Intent.PHYSICS: ExpertConfig(
        intent=Intent.PHYSICS,
        keywords=[
            "fine", "structure", "alpha", "constant", "weinberg", "angle",
            "muon", "electron", "proton", "mass", "ratio", "coupling",
            "electromagnetic", "inverse", "1/alpha", "sin", "theta"
        ]
    ),
    Intent.COSMOLOGY: ExpertConfig(
        intent=Intent.COSMOLOGY,
        keywords=[
            "dark", "matter", "energy", "universe", "cosmic", "hubble",
            "percentage", "fraction", "baryon", "lambda", "cdm", "omega",
            "cosmological", "expansion", "h0", "h naught"
        ]
    ),
    Intent.YANG_MILLS: ExpertConfig(
        intent=Intent.YANG_MILLS,
        keywords=[
            "yang", "mills", "mass", "gap", "qcd", "lambda", "confinement",
            "glueball", "wightman", "axiom", "qft", "gauge", "lattice"
        ]
    ),
    Intent.MIRROR: ExpertConfig(
        intent=Intent.MIRROR,
        keywords=[
            "mirror", "reflect", "symmetry", "transform", "operator",
            "214", "minus", "m(", "apply", "center"
        ]
    ),
    Intent.SEQUENCE: ExpertConfig(
        intent=Intent.SEQUENCE,
        keywords=[
            "sequence", "brahim", "numbers", "list", "b1", "b10",
            "sum", "phi", "dimension", "manifold", "regulator"
        ]
    ),
    Intent.VERIFY: ExpertConfig(
        intent=Intent.VERIFY,
        keywords=[
            "verify", "check", "validate", "prove", "axiom", "satisfied",
            "consistency", "law", "pairs", "sum"
        ]
    ),
    Intent.HELP: ExpertConfig(
        intent=Intent.HELP,
        keywords=[
            "help", "what", "can", "how", "capabilities", "functions",
            "examples", "use", "do", "show"
        ]
    ),
}


@dataclass
class RoutingResult:
    """Result of routing a query through the ephemeral subnet."""
    intent: Intent
    confidence: float
    expert_contributions: Dict[str, float]
    gate_probs: np.ndarray
    selected_experts: List[Intent]
    embedding: np.ndarray
    load_balance_loss: float


class EphemeralOnionSubnet:
    """
    Sparse Mixture of Experts with learned routing through onion layers.

    Architecture:
        Query → Embed → Router → Top-K Experts → Aggregate → Intent

    The "ephemeral" aspect: only top-k experts are activated per query,
    creating a dynamic, sparse computation graph.
    """

    def __init__(self, top_k: int = 2, temperature: float = 0.5,
                 use_keyword_boost: bool = True):
        """
        Initialize ephemeral onion subnet.

        Args:
            top_k: Number of experts to activate per query
            temperature: Softmax temperature (lower = sharper routing)
            use_keyword_boost: Whether to boost routing with keyword matching
        """
        self.top_k = top_k
        self.temperature = temperature
        self.use_keyword_boost = use_keyword_boost

        # Initialize experts
        self.experts: Dict[Intent, ExpertSubnet] = {}
        self.intent_to_idx: Dict[Intent, int] = {}
        self.idx_to_intent: Dict[int, Intent] = {}

        for i, (intent, config) in enumerate(EXPERT_CONFIGS.items()):
            self.experts[intent] = ExpertSubnet(config)
            self.intent_to_idx[intent] = i
            self.idx_to_intent[i] = intent

        # Initialize router
        self.router = GatingNetwork(
            input_dim=DIMENSION,
            num_experts=len(self.experts),
            temperature=temperature
        )

        # Training history
        self.training_history: List[Dict] = []
        self.is_training = True

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed query into 10-dimensional Brahim space.

        Uses combination of:
        1. Keyword-based embedding from expert vocabularies
        2. Character-level features
        3. Structural features (length, punctuation)
        """
        embedding = np.zeros(DIMENSION)
        text_lower = text.lower()

        # 1. Aggregate keyword embeddings from all experts
        keyword_contributions = []
        for intent, expert in self.experts.items():
            for kw, kw_emb in expert.keyword_embeddings.items():
                if kw in text_lower:
                    keyword_contributions.append(kw_emb)

        if keyword_contributions:
            embedding += np.mean(keyword_contributions, axis=0)

        # 2. Character-level features (trigram hash)
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            hash_val = hash(trigram) % DIMENSION
            embedding[hash_val] += 0.1

        # 3. Structural features
        embedding[0] += len(text) / 100.0  # Length feature
        embedding[1] += text.count('?') * 0.5  # Question mark
        embedding[2] += text.count(' ') / 10.0  # Word count proxy

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def route(self, text: str, embedding: Optional[np.ndarray] = None) -> RoutingResult:
        """
        Route query through ephemeral subnet.

        Args:
            text: Input query text
            embedding: Pre-computed embedding (optional)

        Returns:
            RoutingResult with intent, confidence, and expert contributions
        """
        if embedding is None:
            embedding = self.embed_query(text)

        # Get routing probabilities from gate
        gate_probs = self.router.forward(embedding, add_noise=self.is_training)

        # Boost routing with keyword matching
        if self.use_keyword_boost:
            keyword_boost = np.zeros(len(self.experts))
            for intent, expert in self.experts.items():
                idx = self.intent_to_idx[intent]
                keyword_boost[idx] = expert.compute_keyword_similarity(text)

            # Combine gate probs with keyword boost (weighted average)
            alpha = 0.6  # Weight for learned routing
            gate_probs = alpha * gate_probs + (1 - alpha) * keyword_boost
            gate_probs = gate_probs / (np.sum(gate_probs) + 1e-8)

        # Select top-k experts
        top_k_probs, top_k_indices = self.router.select_top_k(gate_probs, self.top_k)

        # Forward through selected experts
        expert_outputs = {}
        for i, idx in enumerate(top_k_indices):
            intent = self.idx_to_intent[idx]
            expert = self.experts[intent]
            confidence = expert.forward(embedding)
            expert_outputs[intent] = top_k_probs[i] * confidence

        # Aggregate: weighted sum of expert confidences
        total_confidence = sum(expert_outputs.values())

        # Select intent with highest contribution
        best_intent = max(expert_outputs.keys(), key=lambda k: expert_outputs[k])

        # Compute load balance loss
        lb_loss = self.router.compute_load_balance_loss()

        return RoutingResult(
            intent=best_intent,
            confidence=total_confidence,
            expert_contributions={k.value: v for k, v in expert_outputs.items()},
            gate_probs=gate_probs,
            selected_experts=[self.idx_to_intent[i] for i in top_k_indices],
            embedding=embedding,
            load_balance_loss=lb_loss
        )

    def train_step(self, text: str, target_intent: Intent, lr: float = 0.01):
        """
        Single training step with query-intent pair.

        Args:
            text: Input query
            target_intent: Ground truth intent
            lr: Learning rate
        """
        embedding = self.embed_query(text)
        result = self.route(text, embedding)

        # Update each expert
        target_idx = self.intent_to_idx[target_intent]
        for intent, expert in self.experts.items():
            idx = self.intent_to_idx[intent]
            target = 1.0 if idx == target_idx else 0.0
            expert.update(embedding, target, lr)

        # Update router with reward signal
        rewards = np.array([
            1.0 if self.idx_to_intent[idx] == target_intent else -0.5
            for idx in [self.intent_to_idx[i] for i in result.selected_experts]
        ])
        selected_indices = np.array([self.intent_to_idx[i] for i in result.selected_experts])
        self.router.update(embedding, selected_indices, rewards, lr)

        # Record history
        self.training_history.append({
            "text": text[:50],
            "target": target_intent.value,
            "predicted": result.intent.value,
            "correct": result.intent == target_intent,
            "confidence": result.confidence,
            "lb_loss": result.load_balance_loss
        })

    def train_batch(self, examples: List[Tuple[str, str]], epochs: int = 5,
                    lr: float = 0.01) -> Dict[str, Any]:
        """
        Train on batch of examples.

        Args:
            examples: List of (text, intent_name) pairs
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            Training statistics
        """
        self.is_training = True

        for epoch in range(epochs):
            np.random.shuffle(examples)
            correct = 0

            for text, intent_name in examples:
                try:
                    target_intent = Intent(intent_name.lower())
                except ValueError:
                    target_intent = Intent.UNKNOWN

                self.train_step(text, target_intent, lr)

                # Check if prediction was correct
                result = self.route(text)
                if result.intent == target_intent:
                    correct += 1

            accuracy = correct / len(examples) if examples else 0

            if epoch % 2 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: Accuracy = {accuracy:.1%}")

        self.is_training = False

        # Compute final statistics
        return {
            "epochs": epochs,
            "examples": len(examples),
            "final_accuracy": accuracy,
            "load_balance_loss": self.router.compute_load_balance_loss(),
            "expert_usage": {
                self.idx_to_intent[i].value: count
                for i, count in enumerate(self.router.expert_usage)
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get subnet statistics."""
        return {
            "num_experts": len(self.experts),
            "top_k": self.top_k,
            "temperature": self.temperature,
            "total_samples": self.router.total_samples,
            "load_balance_loss": self.router.compute_load_balance_loss(),
            "expert_usage": {
                self.idx_to_intent[i].value: int(count)
                for i, count in enumerate(self.router.expert_usage)
            },
            "training_examples": len(self.training_history)
        }


# =============================================================================
# INTEGRATED BOA WITH EPHEMERAL SUBNET
# =============================================================================

class BOAEphemeralAgent:
    """
    Brahim Onion Agent with Ephemeral MoE Subnet.

    Combines:
    - Ephemeral Onion Subnet (learned MoE routing)
    - Brahim calculation execution
    - 3-layer onion response formatting
    """

    def __init__(self, top_k: int = 2, temperature: float = 0.5):
        self.subnet = EphemeralOnionSubnet(top_k=top_k, temperature=temperature)
        self.history: List[Dict] = []

        # Import Brahim SDK functions
        from ..agents_sdk import (
            fine_structure_constant,
            weinberg_angle,
            muon_electron_ratio,
            proton_electron_ratio,
            cosmic_fractions,
            yang_mills_mass_gap,
            mirror_operator,
            get_sequence,
            verify_mirror_symmetry,
        )

        self.calculators = {
            Intent.PHYSICS: self._calc_physics,
            Intent.COSMOLOGY: lambda t: cosmic_fractions().to_dict(),
            Intent.YANG_MILLS: lambda t: yang_mills_mass_gap().to_dict(),
            Intent.MIRROR: self._calc_mirror,
            Intent.SEQUENCE: lambda t: get_sequence(),
            Intent.VERIFY: lambda t: verify_mirror_symmetry(),
            Intent.HELP: self._calc_help,
            Intent.UNKNOWN: self._calc_help,
        }

        self._physics_funcs = {
            "fine": fine_structure_constant,
            "alpha": fine_structure_constant,
            "weinberg": weinberg_angle,
            "muon": muon_electron_ratio,
            "proton": proton_electron_ratio,
        }

    def _calc_physics(self, text: str) -> Dict[str, Any]:
        """Calculate physics constant based on query."""
        from ..agents_sdk import fine_structure_constant

        text_lower = text.lower()
        for key, func in self._physics_funcs.items():
            if key in text_lower:
                return func().to_dict()

        return fine_structure_constant().to_dict()

    def _calc_mirror(self, text: str) -> Dict[str, Any]:
        """Extract number and apply mirror operator."""
        from ..agents_sdk import mirror_operator
        import re

        numbers = re.findall(r'-?\d+', text)
        value = int(numbers[0]) if numbers else 107
        return mirror_operator(value)

    def _calc_help(self, text: str) -> Dict[str, Any]:
        """Return help information."""
        return {
            "message": "Brahim Onion Agent with Ephemeral MoE Subnet",
            "capabilities": list(Intent.__members__.keys()),
            "brahim_sequence": BRAHIM_SEQUENCE,
            "constants": {"S": SUM_CONSTANT, "C": CENTER, "phi": PHI}
        }

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process query through ephemeral subnet.

        Args:
            query: Natural language query

        Returns:
            Response dict with intent, confidence, and calculation result
        """
        if not query or not query.strip():
            return {
                "success": True,
                "intent": "HELP",
                "confidence": 1.0,
                "result": self._calc_help(query)
            }

        # Route through ephemeral subnet
        routing = self.subnet.route(query)

        # Execute calculation
        calc_func = self.calculators.get(routing.intent, self._calc_help)
        calculation = calc_func(query)

        response = {
            "success": True,
            "intent": routing.intent.value.upper(),
            "confidence": routing.confidence,
            "expert_contributions": routing.expert_contributions,
            "selected_experts": [e.value for e in routing.selected_experts],
            "load_balance_loss": routing.load_balance_loss,
            "result": calculation
        }

        self.history.append(response)
        return response

    def train(self, examples: List[Tuple[str, str]], epochs: int = 5) -> Dict[str, Any]:
        """Train the subnet on labeled examples."""
        return self.subnet.train_batch(examples, epochs=epochs)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        subnet_stats = self.subnet.get_stats()
        return {
            "subnet": subnet_stats,
            "history_count": len(self.history),
            "mean_confidence": np.mean([h["confidence"] for h in self.history]) if self.history else 0
        }


# =============================================================================
# TRAINING DATA GENERATOR
# =============================================================================

def generate_training_data() -> List[Tuple[str, str]]:
    """Generate training examples for the ephemeral subnet."""
    return [
        # Physics
        ("What is the fine structure constant?", "physics"),
        ("Calculate alpha", "physics"),
        ("What is the Weinberg angle?", "physics"),
        ("Muon electron mass ratio", "physics"),
        ("What is 1/alpha?", "physics"),
        ("Proton to electron ratio", "physics"),
        ("Electromagnetic coupling constant", "physics"),
        ("What is sin squared theta W?", "physics"),
        ("Calculate the inverse fine structure", "physics"),
        ("What is alpha in physics?", "physics"),

        # Cosmology
        ("What percentage is dark matter?", "cosmology"),
        ("Dark energy fraction", "cosmology"),
        ("How much normal matter in universe?", "cosmology"),
        ("What is the Hubble constant?", "cosmology"),
        ("Cosmic energy budget", "cosmology"),
        ("Baryon fraction of cosmos", "cosmology"),
        ("Give me H naught", "cosmology"),
        ("Dark sector total", "cosmology"),
        ("Omega DM value", "cosmology"),
        ("Lambda CDM parameters", "cosmology"),

        # Yang-Mills
        ("What is the Yang-Mills mass gap?", "yang_mills"),
        ("Calculate Lambda QCD", "yang_mills"),
        ("QCD scale parameter", "yang_mills"),
        ("Wightman axioms satisfied?", "yang_mills"),
        ("Mass gap in MeV", "yang_mills"),
        ("QFT confinement scale", "yang_mills"),
        ("Derive glueball mass", "yang_mills"),
        ("Lattice QCD scale", "yang_mills"),

        # Mirror
        ("Apply mirror to 27", "mirror"),
        ("What is the mirror of 107?", "mirror"),
        ("Transform 75 using mirror operator", "mirror"),
        ("Calculate M(42)", "mirror"),
        ("Reflect 136 through center", "mirror"),
        ("214 minus 97", "mirror"),
        ("Mirror operator on 0", "mirror"),
        ("Apply symmetry transform", "mirror"),

        # Sequence
        ("What is the Brahim sequence?", "sequence"),
        ("List all 10 Brahim numbers", "sequence"),
        ("What is phi in Brahim manifold?", "sequence"),
        ("How many dimensions?", "sequence"),
        ("Give me B1 through B10", "sequence"),
        ("What is the regulator?", "sequence"),
        ("Brahim numbers list", "sequence"),

        # Verify
        ("Verify mirror symmetry", "verify"),
        ("Check Wightman axioms", "verify"),
        ("Are all axioms satisfied?", "verify"),
        ("Validate Brahim laws", "verify"),
        ("Check if pairs sum to 214", "verify"),
        ("Prove consistency relation", "verify"),

        # Help
        ("What can you do?", "help"),
        ("Help me understand capabilities", "help"),
        ("Show me examples", "help"),
        ("How do I use this?", "help"),
        ("What are your functions?", "help"),
    ]


# =============================================================================
# CLI
# =============================================================================

def main():
    """Test the ephemeral onion subnet."""
    print("=" * 70)
    print("EPHEMERAL ONION SUBNET - MIXTURE OF EXPERTS TEST")
    print("=" * 70)

    # Create agent
    agent = BOAEphemeralAgent(top_k=2, temperature=0.5)

    # Generate training data
    training_data = generate_training_data()
    print(f"\nTraining on {len(training_data)} examples...")

    # Train
    train_stats = agent.train(training_data, epochs=10)
    print(f"\nTraining complete!")
    print(f"Final accuracy: {train_stats['final_accuracy']:.1%}")
    print(f"Load balance loss: {train_stats['load_balance_loss']:.4f}")

    # Test queries
    test_queries = [
        "What is the fine structure constant?",
        "Dark matter percentage",
        "Yang-Mills mass gap",
        "Mirror of 75",
        "List the Brahim sequence",
        "Verify axioms",
        "What can you do?",
        "Calculate m_p over m_e",  # Edge case
    ]

    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)

    for query in test_queries:
        result = agent.process(query)
        print(f"\nQ: {query}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Experts: {result['selected_experts']}")

    # Final stats
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    stats = agent.get_stats()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
