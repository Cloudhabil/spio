#!/usr/bin/env python3
"""
Kelimutu Subnet - Three Lakes, One Magma

Inspired by Kelimutu volcano's three crater lakes:
- Surface: Three different colored lakes (visible outputs)
- Underground: Connected through volcanic channels (hidden structure)
- Magma: Single heat source producing all three (unified substrate)

Architecture:
    Query → Magma Substrate (Brahim) → Underground Channels → 3 Lake Outputs

The lakes aren't separate experts - they're THREE EXPRESSIONS of ONE truth,
differentiated by oxidation state (activation function) in the channels.

Coordinates: 8.77°S, 121.82°E (121 ∈ Brahim Sequence)

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import json

# Brahim constants (the magma composition) - Corrected 2026-01-26
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
BRAHIM_SEQUENCE_ORIGINAL = [27, 42, 60, 75, 97, 121, 136, 154, 172, 187]
SUM_CONSTANT = 214  # Pair sum (each mirror pair sums to this)
CENTER = 107        # Equilibrium point
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (crystal structure)
DIMENSION = 10      # Sequence length

# Kelimutu coordinates encode truth
KELIMUTU_LAT = -8.77   # South
KELIMUTU_LON = 121.82  # East - NOTE: ~121 near B₆ (117 in symmetric sequence)


# =============================================================================
# THE THREE LAKES (Output Modes)
# =============================================================================

class Lake(Enum):
    """
    Kelimutu's three crater lakes.

    Each lake has different mineral oxidation → different color
    Same underlying magma → different surface expression
    """
    TIWU_ATA_MBUPU = "old_people"      # Blue-green, stable (verification/axioms)
    TIWU_NUWA_MURI = "young_maidens"   # Turquoise, active (calculation/physics)
    TIWU_ATA_POLO = "enchanted"        # Red-brown, mystic (transformation/mirror)


# Each lake is a PERSPECTIVE, not a category
# All lakes see all intents, but from different angles
ALL_INTENTS = ["physics", "cosmology", "yang_mills", "mirror", "sequence", "verify", "help", "unknown"]

# Lake perspectives:
# - TIWU_ATA_MBUPU (Old People): LITERAL - exact keyword matching
# - TIWU_NUWA_MURI (Young): SEMANTIC - meaning-based inference
# - TIWU_ATA_POLO (Enchanted): STRUCTURAL - pattern/symmetry based


# =============================================================================
# MAGMA SUBSTRATE (Single Source of Truth)
# =============================================================================

class MagmaSubstrate:
    """
    The single heat source beneath all three lakes.

    Contains the Brahim sequence as "mineral composition"
    Provides energy (gradients) to all channels equally
    The TRUTH layer - unified, hidden, fundamental
    """

    def __init__(self):
        # Magma composition = Brahim sequence normalized
        self.composition = np.array(BRAHIM_SEQUENCE) / SUM_CONSTANT

        # Heat capacity (learnable temperature)
        self.temperature = 1.0

        # Crystal structure matrix (encodes relationships)
        self.crystal = self._build_crystal_matrix()

        # Energy state
        self.energy = 0.0

    def _build_crystal_matrix(self) -> np.ndarray:
        """
        Build crystal structure from Brahim sequence.

        Encodes mirror symmetry: M(x) = 214 - x
        And golden ratio relationships
        """
        crystal = np.zeros((DIMENSION, DIMENSION))

        for i in range(DIMENSION):
            for j in range(DIMENSION):
                # Mirror pair relationship
                bi, bj = BRAHIM_SEQUENCE[i], BRAHIM_SEQUENCE[j]
                mirror_i = SUM_CONSTANT - bi

                # Check if j is mirror of i
                if bj == mirror_i:
                    crystal[i, j] = 1.0  # Strong bond
                elif abs(bi - bj) == abs(BRAHIM_SEQUENCE[1] - BRAHIM_SEQUENCE[0]):
                    crystal[i, j] = PHI / 10  # Sequential bond
                else:
                    # Distance-based weak bond
                    crystal[i, j] = 1.0 / (1 + abs(bi - bj) / CENTER)

        # Normalize rows
        row_sums = crystal.sum(axis=1, keepdims=True)
        crystal = crystal / (row_sums + 1e-8)

        return crystal

    def heat(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Apply magma heat to query embedding.

        The heat transforms the query through the crystal structure,
        preparing it for differentiation in the channels.
        """
        # Project through crystal structure
        heated = query_embedding @ self.crystal

        # Apply temperature scaling
        heated = heated * self.temperature

        # Compute energy contribution
        self.energy = np.sum(heated ** 2)

        return heated

    def get_mineral_signature(self, text: str) -> np.ndarray:
        """
        Extract mineral signature from text.

        Maps text to Brahim space based on keyword presence.
        """
        signature = np.zeros(DIMENSION)
        text_lower = text.lower()

        # Each Brahim number corresponds to certain concepts
        mineral_keywords = {
            0: ["first", "initial", "start", "begin", "27"],
            1: ["second", "ratio", "proportion", "42"],
            2: ["third", "angle", "geometry", "60"],
            3: ["fourth", "mass", "matter", "75"],
            4: ["fifth", "prime", "fundamental", "97"],
            5: ["sixth", "energy", "force", "121"],  # 121 = Kelimutu longitude!
            6: ["seventh", "transform", "change", "136"],
            7: ["eighth", "symmetry", "mirror", "154"],
            8: ["ninth", "cosmos", "universe", "172"],
            9: ["tenth", "complete", "total", "187"],
        }

        for idx, keywords in mineral_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    signature[idx] += self.composition[idx]

        # Add character-level features
        for i, char in enumerate(text_lower[:DIMENSION]):
            signature[i % DIMENSION] += ord(char) / 1000

        # Normalize
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm

        return signature


# =============================================================================
# UNDERGROUND CHANNELS (Hidden Connections)
# =============================================================================

class UndergroundChannel:
    """
    Volcanic channel connecting magma to a specific lake.

    Each channel has:
    - Different mineral filter (what passes through)
    - Different oxidation function (how it transforms)
    - Connection to other channels (lateral flow)
    """

    def __init__(self, lake: Lake, oxidation_bias: float = 0.0):
        self.lake = lake
        self.oxidation_bias = oxidation_bias

        # Channel-specific filter (learnable)
        self.mineral_filter = np.random.randn(DIMENSION) * 0.1
        self.mineral_filter += self._get_lake_affinity()

        # Lateral connections to other channels
        self.lateral_weights = {}

        # Flow statistics
        self.total_flow = 0.0
        self.activation_count = 0

    def _get_lake_affinity(self) -> np.ndarray:
        """Get initial mineral affinity based on lake type."""
        affinity = np.zeros(DIMENSION)

        if self.lake == Lake.TIWU_ATA_MBUPU:  # Old People - wisdom
            # Affinity for higher, stable numbers
            affinity[7:] = 0.3  # 154, 172, 187

        elif self.lake == Lake.TIWU_NUWA_MURI:  # Young - energy
            # Affinity for middle, dynamic numbers
            affinity[3:7] = 0.3  # 75, 97, 121, 136

        elif self.lake == Lake.TIWU_ATA_POLO:  # Enchanted - transformation
            # Affinity for symmetric pairs
            affinity[0] = 0.3  # 27 ↔ 187
            affinity[9] = 0.3
            affinity[4] = 0.3  # 97 ↔ 117 (close to center)

        return affinity

    def oxidize(self, heated_embedding: np.ndarray) -> float:
        """
        Apply oxidation transformation.

        Different lakes have different oxidation states:
        - Blue lake: Low oxidation (Fe²⁺)
        - Green lake: Medium oxidation
        - Red lake: High oxidation (Fe³⁺)

        Returns activation level for this channel.
        """
        # Apply mineral filter
        filtered = heated_embedding * self.mineral_filter

        # Oxidation transformation (different activation per lake)
        if self.lake == Lake.TIWU_ATA_MBUPU:
            # Stable, sigmoid-like (wisdom needs certainty)
            activation = 1 / (1 + np.exp(-np.sum(filtered) + self.oxidation_bias))

        elif self.lake == Lake.TIWU_NUWA_MURI:
            # Dynamic, tanh-like (energy oscillates)
            activation = (np.tanh(np.sum(filtered) + self.oxidation_bias) + 1) / 2

        elif self.lake == Lake.TIWU_ATA_POLO:
            # Mystic, softplus-like (transformation is gradual)
            x = np.sum(filtered) + self.oxidation_bias
            activation = np.log(1 + np.exp(x)) / 3  # Scaled softplus

        else:
            activation = np.sum(filtered)

        self.total_flow += activation
        self.activation_count += 1

        return float(np.clip(activation, 0, 1))

    def connect_lateral(self, other_channel: 'UndergroundChannel', weight: float):
        """Establish lateral connection to another channel."""
        self.lateral_weights[other_channel.lake] = weight

    def receive_lateral_flow(self, flows: Dict[Lake, float]) -> float:
        """Receive flow from connected channels."""
        lateral_contribution = 0.0
        for lake, weight in self.lateral_weights.items():
            if lake in flows:
                lateral_contribution += weight * flows[lake]
        return lateral_contribution

    def update_filter(self, gradient: np.ndarray, lr: float = 0.01):
        """Update mineral filter based on gradient."""
        self.mineral_filter += lr * gradient


# =============================================================================
# KELIMUTU SUBNET (Complete System)
# =============================================================================

@dataclass
class KelimutuOutput:
    """Output from Kelimutu subnet routing."""
    intent: str
    confidence: float
    lake: Lake
    lake_activations: Dict[str, float]
    underground_flow: float
    magma_energy: float
    mineral_signature: np.ndarray


class KelimutuSubnet:
    """
    Three Lakes, One Magma - Three Perspectives, One Truth.

    Each lake provides a DIFFERENT PERSPECTIVE on ALL intents:
    - Lake 1 (Old People): LITERAL - keyword matching
    - Lake 2 (Young Maidens): SEMANTIC - meaning inference
    - Lake 3 (Enchanted): STRUCTURAL - pattern recognition

    The underground channels share information between perspectives.
    Final intent comes from FUSION of all three views.
    """

    def __init__(self):
        # The single source of truth
        self.magma = MagmaSubstrate()

        # Three perspective channels
        self.channels = {
            Lake.TIWU_ATA_MBUPU: UndergroundChannel(Lake.TIWU_ATA_MBUPU, oxidation_bias=-0.5),
            Lake.TIWU_NUWA_MURI: UndergroundChannel(Lake.TIWU_NUWA_MURI, oxidation_bias=0.0),
            Lake.TIWU_ATA_POLO: UndergroundChannel(Lake.TIWU_ATA_POLO, oxidation_bias=0.3),
        }

        # Establish underground connections
        self._connect_channels()

        # Each lake has its own intent weights (different perspective)
        self.lake_intent_weights = {
            lake: self._init_intent_weights_for_lake(lake)
            for lake in Lake
        }

        # Fusion weights (how much to trust each lake)
        self.fusion_weights = {
            Lake.TIWU_ATA_MBUPU: 0.4,   # Literal: high trust for exact matches
            Lake.TIWU_NUWA_MURI: 0.35,  # Semantic: medium trust
            Lake.TIWU_ATA_POLO: 0.25,   # Structural: lower but catches edge cases
        }

        # Keyword database for literal matching
        self.keywords = self._init_keywords()

        self.is_training = True
        self.history = []

    def _connect_channels(self):
        """Establish underground connections between channels."""
        for lake1, channel1 in self.channels.items():
            for lake2, channel2 in self.channels.items():
                if lake1 != lake2:
                    channel1.connect_lateral(channel2, 0.15)

    def _init_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword database for literal matching."""
        return {
            "physics": ["fine", "structure", "alpha", "constant", "weinberg", "angle",
                       "muon", "electron", "proton", "mass", "ratio", "coupling",
                       "electromagnetic", "1/alpha", "sin", "theta"],
            "cosmology": ["dark", "matter", "energy", "universe", "cosmic", "hubble",
                         "percentage", "fraction", "baryon", "lambda", "cdm", "omega",
                         "cosmological", "expansion", "h0", "h naught"],
            "yang_mills": ["yang", "mills", "mass", "gap", "qcd", "confinement",
                          "glueball", "wightman", "qft", "gauge", "lattice", "lambda qcd"],
            "mirror": ["mirror", "reflect", "symmetry", "transform", "operator",
                      "214", "minus", "m(", "apply", "center", "214 -"],
            "sequence": ["sequence", "brahim", "numbers", "list", "b1", "b10",
                        "sum", "phi", "dimension", "manifold", "regulator"],
            "verify": ["verify", "check", "validate", "prove", "axiom", "satisfied",
                      "consistency", "law", "pairs", "214"],
            "help": ["help", "what can", "how", "capabilities", "functions",
                    "examples", "use", "do", "show"],
            "unknown": []
        }

    def _init_intent_weights_for_lake(self, lake: Lake) -> Dict[str, np.ndarray]:
        """Initialize intent weights specific to each lake's perspective."""
        weights = {}

        for intent in ALL_INTENTS:
            w = np.random.randn(DIMENSION) * 0.1

            if lake == Lake.TIWU_ATA_MBUPU:
                # Literal: strong keyword-position correlation
                if intent == "physics":
                    w[3:6] += 0.4
                elif intent == "cosmology":
                    w[8] += 0.5
                elif intent == "mirror":
                    w[7] += 0.5
                elif intent == "sequence":
                    w[:3] += 0.3

            elif lake == Lake.TIWU_NUWA_MURI:
                # Semantic: broader, meaning-based patterns
                if intent in ["physics", "cosmology", "yang_mills"]:
                    w[3:8] += 0.2  # Science-related positions
                elif intent in ["mirror", "verify"]:
                    w[6:9] += 0.2  # Symmetry positions

            elif lake == Lake.TIWU_ATA_POLO:
                # Structural: pattern and symmetry based
                # Emphasize mirror pairs in Brahim sequence
                w[0] += 0.1
                w[9] += 0.1  # 27 <-> 187
                w[4] += 0.1
                w[5] += 0.1  # 97 <-> 117 area

            weights[intent] = w

        return weights

    def _literal_classify(self, text: str) -> Dict[str, float]:
        """Lake 1: Literal keyword matching."""
        scores = {intent: 0.0 for intent in ALL_INTENTS}
        text_lower = text.lower()

        for intent, keywords in self.keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1.0
                    # Bonus for multi-word matches
                    if ' ' in kw and kw in text_lower:
                        scores[intent] += 0.5

        # Normalize
        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}

    def _semantic_classify(self, mineral_sig: np.ndarray, lake: Lake) -> Dict[str, float]:
        """Lake 2: Semantic meaning-based classification."""
        scores = {}
        weights = self.lake_intent_weights[lake]

        for intent, w in weights.items():
            alignment = float(np.dot(mineral_sig, w))
            scores[intent] = max(0, alignment + 0.5)

        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}

    def _structural_classify(self, mineral_sig: np.ndarray, heated: np.ndarray) -> Dict[str, float]:
        """Lake 3: Structural pattern-based classification."""
        scores = {intent: 0.0 for intent in ALL_INTENTS}

        # Check for mirror symmetry patterns
        symmetry_score = 0.0
        for i in range(DIMENSION // 2):
            symmetry_score += abs(mineral_sig[i] - mineral_sig[DIMENSION - 1 - i])
        symmetry_score = 1.0 - (symmetry_score / (DIMENSION // 2))

        # High symmetry suggests mirror/verify intents
        scores["mirror"] += symmetry_score * 0.5
        scores["verify"] += symmetry_score * 0.3

        # Energy concentration suggests physics/cosmology
        energy_concentration = np.max(heated) / (np.mean(heated) + 1e-8)
        if energy_concentration > 2:
            scores["physics"] += 0.3
            scores["yang_mills"] += 0.2

        # Uniform distribution suggests sequence/help
        uniformity = 1.0 - np.std(mineral_sig)
        scores["sequence"] += uniformity * 0.3
        scores["help"] += uniformity * 0.2

        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}

    def route(self, text: str) -> KelimutuOutput:
        """
        Route through three lakes and fuse results.

        Each lake provides scores for ALL intents.
        Underground connections share information.
        Final answer is weighted fusion of all perspectives.
        """
        # Extract mineral signature
        mineral_sig = self.magma.get_mineral_signature(text)
        heated = self.magma.heat(mineral_sig)

        # Get channel activations
        channel_activations = {}
        for lake, channel in self.channels.items():
            channel_activations[lake] = channel.oxidize(heated)

        # Each lake classifies ALL intents from its perspective
        lake_scores = {
            Lake.TIWU_ATA_MBUPU: self._literal_classify(text),
            Lake.TIWU_NUWA_MURI: self._semantic_classify(mineral_sig, Lake.TIWU_NUWA_MURI),
            Lake.TIWU_ATA_POLO: self._structural_classify(mineral_sig, heated),
        }

        # Fuse: weighted combination of all lake perspectives
        fused_scores = {intent: 0.0 for intent in ALL_INTENTS}

        for lake, scores in lake_scores.items():
            lake_weight = self.fusion_weights[lake]
            channel_activation = channel_activations[lake]

            for intent, score in scores.items():
                # Weight by both fusion weight and channel activation
                fused_scores[intent] += lake_weight * channel_activation * score

        # Select best intent
        best_intent = max(fused_scores.keys(), key=lambda k: fused_scores[k])

        # Confidence = score of best / sum of all
        total_score = sum(fused_scores.values()) + 1e-8
        confidence = fused_scores[best_intent] / total_score

        # Determine which lake contributed most
        lake_contributions = {}
        for lake, scores in lake_scores.items():
            lake_contributions[lake] = scores.get(best_intent, 0) * self.fusion_weights[lake]
        dominant_lake = max(lake_contributions.keys(), key=lambda k: lake_contributions[k])

        return KelimutuOutput(
            intent=best_intent,
            confidence=confidence,
            lake=dominant_lake,
            lake_activations={l.value: a for l, a in channel_activations.items()},
            underground_flow=sum(channel_activations.values()),
            magma_energy=self.magma.energy,
            mineral_signature=mineral_sig
        )

    def train_step(self, text: str, target_intent: str, lr: float = 0.01):
        """Single training step - update all lake weights."""
        result = self.route(text)
        mineral_sig = result.mineral_signature

        # Update weights for each lake
        for lake, weights in self.lake_intent_weights.items():
            if target_intent in weights:
                # Increase weight for correct intent
                weights[target_intent] += lr * mineral_sig

            if result.intent != target_intent and result.intent in weights:
                # Decrease weight for wrong prediction
                weights[result.intent] -= lr * 0.5 * mineral_sig

        # Update fusion weights based on which lake was correct
        for lake, scores in [(Lake.TIWU_ATA_MBUPU, self._literal_classify(text)),
                             (Lake.TIWU_NUWA_MURI, self._semantic_classify(mineral_sig, Lake.TIWU_NUWA_MURI)),
                             (Lake.TIWU_ATA_POLO, self._structural_classify(mineral_sig, result.mineral_signature))]:
            lake_prediction = max(scores.keys(), key=lambda k: scores[k])
            if lake_prediction == target_intent:
                # This lake was right, increase its fusion weight
                self.fusion_weights[lake] = min(0.6, self.fusion_weights[lake] + lr * 0.1)
            else:
                # This lake was wrong, decrease slightly
                self.fusion_weights[lake] = max(0.1, self.fusion_weights[lake] - lr * 0.05)

        # Renormalize fusion weights
        total_fusion = sum(self.fusion_weights.values())
        self.fusion_weights = {k: v / total_fusion for k, v in self.fusion_weights.items()}

        self.history.append({
            "text": text[:30],
            "target": target_intent,
            "predicted": result.intent,
            "correct": result.intent == target_intent
        })

    def train(self, examples: List[Tuple[str, str]], epochs: int = 10,
              lr: float = 0.01) -> Dict[str, Any]:
        """Train on examples."""
        self.is_training = True

        for epoch in range(epochs):
            np.random.shuffle(examples)
            correct = 0

            for text, intent in examples:
                self.train_step(text, intent, lr)
                result = self.route(text)
                if result.intent == intent:
                    correct += 1

            acc = correct / len(examples)
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: {acc:.1%}")

        self.is_training = False

        return {
            "epochs": epochs,
            "final_accuracy": acc,
            "examples": len(examples),
            "fusion_weights": {k.value: v for k, v in self.fusion_weights.items()}
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "magma_temperature": self.magma.temperature,
            "magma_energy": self.magma.energy,
            "fusion_weights": {k.value: v for k, v in self.fusion_weights.items()},
            "channel_flows": {
                lake.value: {
                    "total_flow": ch.total_flow,
                    "activations": ch.activation_count
                }
                for lake, ch in self.channels.items()
            },
            "training_samples": len(self.history)
        }


# =============================================================================
# BOA KELIMUTU AGENT
# =============================================================================

class BOAKelimutuAgent:
    """
    Brahim Onion Agent with Kelimutu Subnet.

    Three lakes, one magma, hidden underground connections.
    """

    def __init__(self):
        self.subnet = KelimutuSubnet()
        self.history = []

        # Import calculators
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

        self._calcs = {
            "physics": self._calc_physics,
            "cosmology": lambda t: cosmic_fractions().to_dict(),
            "yang_mills": lambda t: yang_mills_mass_gap().to_dict(),
            "mirror": self._calc_mirror,
            "sequence": lambda t: get_sequence(),
            "verify": lambda t: verify_mirror_symmetry(),
            "help": self._help,
            "unknown": self._help,
        }

        self._physics = {
            "fine": fine_structure_constant,
            "alpha": fine_structure_constant,
            "weinberg": weinberg_angle,
            "muon": muon_electron_ratio,
            "proton": proton_electron_ratio,
        }

    def _calc_physics(self, text: str) -> Dict:
        from ..agents_sdk import fine_structure_constant
        for k, f in self._physics.items():
            if k in text.lower():
                return f().to_dict()
        return fine_structure_constant().to_dict()

    def _calc_mirror(self, text: str) -> Dict:
        from ..agents_sdk import mirror_operator
        import re
        nums = re.findall(r'-?\d+', text)
        val = int(nums[0]) if nums else 107
        return mirror_operator(val)

    def _help(self, text: str) -> Dict:
        return {
            "message": "Kelimutu Subnet - Three Lakes, One Magma",
            "lakes": {
                "Tiwu Ata Mbupu": "Wisdom (verify, help, sequence)",
                "Tiwu Nuwa Muri": "Energy (physics, cosmology, yang_mills)",
                "Tiwu Ata Polo": "Transformation (mirror)"
            },
            "coordinates": {"lat": KELIMUTU_LAT, "lon": KELIMUTU_LON},
            "brahim_sequence": BRAHIM_SEQUENCE
        }

    def process(self, query: str) -> Dict[str, Any]:
        """Process query through Kelimutu subnet."""
        if not query.strip():
            return {"success": True, "intent": "HELP", "result": self._help("")}

        # Route through subnet
        routing = self.subnet.route(query)

        # Execute calculation
        calc = self._calcs.get(routing.intent, self._help)
        result = calc(query)

        response = {
            "success": True,
            "intent": routing.intent.upper(),
            "confidence": routing.confidence,
            "lake": routing.lake.value,
            "lake_activations": routing.lake_activations,
            "underground_flow": routing.underground_flow,
            "magma_energy": routing.magma_energy,
            "result": result
        }

        self.history.append(response)
        return response

    def train(self, examples: List[Tuple[str, str]], epochs: int = 10) -> Dict:
        """Train the subnet."""
        return self.subnet.train(examples, epochs=epochs)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "subnet": self.subnet.get_stats(),
            "history_count": len(self.history)
        }


# =============================================================================
# CLI TEST
# =============================================================================

def main():
    print("=" * 70)
    print("KELIMUTU SUBNET - THREE LAKES, ONE MAGMA")
    print(f"Coordinates: {KELIMUTU_LAT}°S, {KELIMUTU_LON}°E")
    print(f"Note: Longitude 121.82 ≈ B₆ = 121 in Brahim sequence")
    print("=" * 70)

    agent = BOAKelimutuAgent()

    # Training data
    from .ephemeral_subnet import generate_training_data
    training_data = generate_training_data()

    print(f"\nTraining on {len(training_data)} examples...")
    stats = agent.train(training_data, epochs=10)
    print(f"Final accuracy: {stats['final_accuracy']:.1%}")

    # Test queries
    tests = [
        "What is the fine structure constant?",
        "Dark matter percentage",
        "Yang-Mills mass gap",
        "Mirror of 75",
        "Brahim sequence",
        "Verify axioms",
    ]

    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)

    for q in tests:
        r = agent.process(q)
        print(f"\nQ: {q}")
        print(f"  Lake: {r['lake']}")
        print(f"  Intent: {r['intent']} ({r['confidence']:.1%})")
        print(f"  Flow: {r['underground_flow']:.3f}")


if __name__ == "__main__":
    main()
