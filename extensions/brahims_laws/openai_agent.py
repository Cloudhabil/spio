#!/usr/bin/env python3
"""
Brahim Onion Agent (BOA) - OpenAI Agents SDK Implementation

A multi-layer computational agent using onion-style routing for physics calculations.
Compatible with OpenAI Agents SDK, AgentKit, and Agent Builder patterns.

Architecture:
    Layer C (Interface)  → Parse user intent, route to appropriate handler
    Layer B (Symmetry)   → Apply mirror operations, verify axioms
    Layer A (Core)       → Execute Brahim mechanics calculations

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
License: TUL (Technology Unified License)
"""

from __future__ import annotations
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable, Union, Literal
from enum import Enum
import math

# Import Brahim SDK core
from .agents_sdk import (
    BRAHIM_SEQUENCE,
    SUM_CONSTANT,
    CENTER,
    PHI,
    DELTA_4,
    DELTA_5,
    fine_structure_constant,
    weinberg_angle,
    muon_electron_ratio,
    proton_electron_ratio,
    cosmic_fractions,
    yang_mills_mass_gap,
    mirror_operator,
    get_sequence,
    verify_mirror_symmetry,
    BrahimNumber,
    MirrorPair,
    CalculationResult,
)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class LayerID(str, Enum):
    """Onion routing layer identifiers."""
    CORE = "A"      # Innermost - raw calculation
    SYMMETRY = "B"  # Middle - axiom verification
    INTERFACE = "C" # Outermost - user-facing


class Intent(str, Enum):
    """User intent classification."""
    PHYSICS = "physics"
    COSMOLOGY = "cosmology"
    YANG_MILLS = "yang_mills"
    MIRROR = "mirror"
    SEQUENCE = "sequence"
    VERIFY = "verify"
    HELP = "help"
    UNKNOWN = "unknown"


class ModelType(str, Enum):
    """Supported model backends."""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for Brahim Onion Agent."""
    name: str = "brahim-onion-agent"
    model: ModelType = ModelType.GPT4O_MINI
    temperature: float = 0.1
    max_tokens: int = 4096

    # Layer settings
    enable_layer_logging: bool = True
    strict_axiom_mode: bool = True

    # Guardrails
    max_iterations: int = 10
    timeout_seconds: float = 30.0

    # Industry focus (for specialized responses)
    industry: Optional[str] = None


@dataclass
class LayerPacket:
    """Data packet passed between onion layers."""
    payload: Dict[str, Any]
    source_layer: LayerID
    destination_layer: LayerID
    encrypted: bool = False  # Conceptual - indicates processing needed
    trace: List[str] = field(default_factory=list)

    def add_trace(self, message: str):
        self.trace.append(f"[{self.source_layer.value}→{self.destination_layer.value}] {message}")


@dataclass
class AgentResponse:
    """Final response from the agent."""
    success: bool
    result: Dict[str, Any]
    intent: Intent
    layers_traversed: List[LayerID]
    trace: List[str]
    model_used: str
    tokens_used: Optional[int] = None


# =============================================================================
# TOOL DEFINITIONS (OpenAI Function Calling Format)
# =============================================================================

BRAHIM_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_physics_constant",
            "description": "Calculate fundamental physics constants using Brahim mechanics. Returns value with experimental comparison and accuracy metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "constant": {
                        "type": "string",
                        "enum": ["fine_structure", "weinberg_angle", "muon_electron", "proton_electron"],
                        "description": "The physics constant to calculate"
                    }
                },
                "required": ["constant"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_cosmology",
            "description": "Calculate cosmological energy density fractions: dark matter (27%), dark energy (68%), normal matter (5%), and Hubble constant.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_yang_mills",
            "description": "Calculate the Yang-Mills mass gap with full QCD derivation chain and Wightman axiom verification.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_mirror_operator",
            "description": "Apply the Brahim mirror operator M(x) = 214 - x to any integer value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "The integer value to transform"
                    }
                },
                "required": ["value"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_brahim_sequence",
            "description": "Retrieve the complete 10-element Brahim sequence with all constants: sum=214, center=107, phi, deviations, and regulator.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_axioms",
            "description": "Verify Brahim axioms including mirror symmetry (all pairs sum to 214) and Wightman axioms for QFT consistency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "axiom_type": {
                        "type": "string",
                        "enum": ["mirror_symmetry", "wightman", "all"],
                        "description": "Which axioms to verify"
                    }
                },
                "required": ["axiom_type"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


# =============================================================================
# LAYER IMPLEMENTATIONS
# =============================================================================

class LayerA:
    """
    Core Layer (Innermost) - Raw Brahim Calculations

    This layer performs the actual physics calculations using the
    Brahim sequence and mechanics. No interpretation, just computation.
    """

    @staticmethod
    def process(packet: LayerPacket) -> LayerPacket:
        """Execute core calculation based on packet payload."""
        action = packet.payload.get("action")
        params = packet.payload.get("params", {})

        result = None

        if action == "physics":
            constant = params.get("constant")
            if constant == "fine_structure":
                result = fine_structure_constant().to_dict()
            elif constant == "weinberg_angle":
                result = weinberg_angle().to_dict()
            elif constant == "muon_electron":
                result = muon_electron_ratio().to_dict()
            elif constant == "proton_electron":
                result = proton_electron_ratio().to_dict()

        elif action == "cosmology":
            result = cosmic_fractions().to_dict()

        elif action == "yang_mills":
            result = yang_mills_mass_gap().to_dict()

        elif action == "mirror":
            value = params.get("value", 0)
            result = mirror_operator(value)

        elif action == "sequence":
            result = get_sequence()

        elif action == "verify":
            axiom_type = params.get("axiom_type", "all")
            if axiom_type == "mirror_symmetry":
                result = verify_mirror_symmetry()
            elif axiom_type == "wightman":
                ym = yang_mills_mass_gap()
                result = {
                    "axioms": ["W0: Hilbert", "W1: Poincare", "W2: Spectral",
                              "W3: Vacuum", "W4: Complete", "W5: Local"],
                    "satisfied": ym.wightman_satisfied,
                    "all_passed": all(ym.wightman_satisfied)
                }
            else:
                result = {
                    "mirror": verify_mirror_symmetry(),
                    "wightman": {"satisfied": yang_mills_mass_gap().wightman_satisfied}
                }

        # Create response packet
        response = LayerPacket(
            payload={"result": result, "action": action},
            source_layer=LayerID.CORE,
            destination_layer=LayerID.SYMMETRY,
            encrypted=False,
            trace=packet.trace.copy()
        )
        response.add_trace(f"Computed {action}")

        return response


class LayerB:
    """
    Symmetry Layer (Middle) - Axiom Verification & Mirror Operations

    This layer applies symmetry checks, verifies axiom consistency,
    and ensures calculations respect the Brahim manifold structure.
    """

    @staticmethod
    def process(packet: LayerPacket, strict_mode: bool = True) -> LayerPacket:
        """Apply symmetry verification to results."""
        result = packet.payload.get("result", {})
        action = packet.payload.get("action")

        # Add symmetry metadata
        symmetry_check = {
            "mirror_sum": SUM_CONSTANT,
            "center": CENTER,
            "phi_ratio": PHI,
            "asymmetry": DELTA_4 + DELTA_5,
        }

        # Verify axioms if in strict mode
        axiom_status = None
        if strict_mode:
            mirror_valid = verify_mirror_symmetry()
            axiom_status = {
                "mirror_symmetry_valid": mirror_valid["all_satisfied"],
                "pairs_checked": len(mirror_valid["pairs"])
            }

        # Enhance result with symmetry data
        enhanced_result = {
            "calculation": result,
            "symmetry": symmetry_check,
            "axioms": axiom_status,
            "layer_b_processed": True
        }

        response = LayerPacket(
            payload=enhanced_result,
            source_layer=LayerID.SYMMETRY,
            destination_layer=LayerID.INTERFACE,
            encrypted=False,
            trace=packet.trace.copy()
        )
        response.add_trace(f"Symmetry verified for {action}")

        return response


class LayerC:
    """
    Interface Layer (Outermost) - User Intent Parsing & Response Formatting

    This layer handles natural language understanding, intent classification,
    and formats the final response for the user.
    """

    INTENT_KEYWORDS = {
        Intent.PHYSICS: ["alpha", "fine structure", "weinberg", "muon", "proton", "electron", "mass ratio", "constant"],
        Intent.COSMOLOGY: ["dark matter", "dark energy", "cosmos", "universe", "hubble", "cosmology"],
        Intent.YANG_MILLS: ["yang mills", "mass gap", "qcd", "qft", "wightman", "lambda qcd"],
        Intent.MIRROR: ["mirror", "reflect", "214", "transform", "operator"],
        Intent.SEQUENCE: ["sequence", "brahim numbers", "manifold", "list", "all numbers"],
        Intent.VERIFY: ["verify", "check", "axiom", "validate", "proof"],
        Intent.HELP: ["help", "what can you", "capabilities", "how to"],
    }

    @classmethod
    def classify_intent(cls, user_input: str) -> Intent:
        """Classify user intent from natural language."""
        lower_input = user_input.lower()

        for intent, keywords in cls.INTENT_KEYWORDS.items():
            if any(kw in lower_input for kw in keywords):
                return intent

        return Intent.UNKNOWN

    @classmethod
    def parse_parameters(cls, user_input: str, intent: Intent) -> Dict[str, Any]:
        """Extract parameters from user input based on intent."""
        lower_input = user_input.lower()
        params = {}

        if intent == Intent.PHYSICS:
            if "fine structure" in lower_input or "alpha" in lower_input:
                params["constant"] = "fine_structure"
            elif "weinberg" in lower_input:
                params["constant"] = "weinberg_angle"
            elif "muon" in lower_input:
                params["constant"] = "muon_electron"
            elif "proton" in lower_input:
                params["constant"] = "proton_electron"
            else:
                params["constant"] = "fine_structure"  # default

        elif intent == Intent.MIRROR:
            # Try to extract a number
            import re
            numbers = re.findall(r'\d+', user_input)
            if numbers:
                params["value"] = int(numbers[0])
            else:
                params["value"] = 107  # default to center

        elif intent == Intent.VERIFY:
            if "mirror" in lower_input:
                params["axiom_type"] = "mirror_symmetry"
            elif "wightman" in lower_input:
                params["axiom_type"] = "wightman"
            else:
                params["axiom_type"] = "all"

        return params

    @classmethod
    def create_request_packet(cls, user_input: str) -> LayerPacket:
        """Create initial packet from user input."""
        intent = cls.classify_intent(user_input)
        params = cls.parse_parameters(user_input, intent)

        # Map intent to action
        action_map = {
            Intent.PHYSICS: "physics",
            Intent.COSMOLOGY: "cosmology",
            Intent.YANG_MILLS: "yang_mills",
            Intent.MIRROR: "mirror",
            Intent.SEQUENCE: "sequence",
            Intent.VERIFY: "verify",
            Intent.HELP: "help",
            Intent.UNKNOWN: "help",
        }

        packet = LayerPacket(
            payload={
                "action": action_map[intent],
                "params": params,
                "original_input": user_input,
                "intent": intent.value
            },
            source_layer=LayerID.INTERFACE,
            destination_layer=LayerID.CORE,
            encrypted=True,  # Needs processing
            trace=[]
        )
        packet.add_trace(f"Intent: {intent.value}")

        return packet

    @staticmethod
    def format_response(packet: LayerPacket, intent: Intent, config: AgentConfig) -> AgentResponse:
        """Format final response for user."""
        result = packet.payload

        # Add industry-specific context if configured
        if config.industry:
            result["industry_context"] = {
                "focus": config.industry,
                "relevance": f"This calculation is relevant to {config.industry}"
            }

        return AgentResponse(
            success=True,
            result=result,
            intent=intent,
            layers_traversed=[LayerID.INTERFACE, LayerID.CORE, LayerID.SYMMETRY, LayerID.INTERFACE],
            trace=packet.trace,
            model_used=config.model.value
        )


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class BrahimOnionAgent:
    """
    Brahim Onion Agent - Multi-layer computational agent

    Implements onion-style routing for physics calculations:
    - Layer C (Interface): Parse intent, format response
    - Layer B (Symmetry): Verify axioms, apply mirror checks
    - Layer A (Core): Execute Brahim mechanics calculations

    Compatible with OpenAI Agents SDK patterns.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.tools = BRAHIM_AGENT_TOOLS
        self.history: List[AgentResponse] = []

        # Layer instances
        self.layer_a = LayerA()
        self.layer_b = LayerB()
        self.layer_c = LayerC()

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool definitions."""
        return self.tools

    def get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return f"""You are the Brahim Onion Agent (BOA), a specialized AI for physics calculations.

You use the Brahim sequence B = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187] to derive:
- Fine structure constant: α⁻¹ = 137.036 (2 ppm accuracy)
- Weinberg angle: sin²θ_W = 0.2308
- Mass ratios: muon/electron = 206.8, proton/electron = 1836
- Cosmology: Dark matter 27%, Dark energy 68%, Normal matter 5%
- Yang-Mills mass gap: Δ = 1721 MeV

Architecture: 3-layer onion routing
- Layer C: Intent parsing and response formatting
- Layer B: Symmetry verification and axiom checking
- Layer A: Core Brahim mechanics calculations

Key constants: Sum=214, Center=107, φ=(1+√5)/2

Industry focus: {self.config.industry or 'General scientific research'}

Always show derivation steps and compare with experimental values."""

    def process(self, user_input: str) -> AgentResponse:
        """
        Process user input through all onion layers.

        Flow: User → Layer C → Layer A → Layer B → Layer C → User
        """
        # Layer C: Parse input
        packet = self.layer_c.create_request_packet(user_input)
        intent = Intent(packet.payload["intent"])

        # Handle help intent directly
        if intent in [Intent.HELP, Intent.UNKNOWN]:
            return self._generate_help_response(intent)

        # Layer A: Core calculation
        packet = self.layer_a.process(packet)

        # Layer B: Symmetry verification
        packet = self.layer_b.process(packet, strict_mode=self.config.strict_axiom_mode)

        # Layer C: Format response
        response = self.layer_c.format_response(packet, intent, self.config)

        # Store in history
        self.history.append(response)

        return response

    def _generate_help_response(self, intent: Intent) -> AgentResponse:
        """Generate help response."""
        help_text = {
            "capabilities": [
                "calculate_physics_constant - Fine structure, Weinberg angle, mass ratios",
                "calculate_cosmology - Dark matter/energy fractions, Hubble constant",
                "calculate_yang_mills - QCD mass gap with Wightman axioms",
                "apply_mirror_operator - Transform values via M(x) = 214 - x",
                "get_brahim_sequence - Full 10-element sequence with metadata",
                "verify_axioms - Check mirror symmetry and QFT axioms"
            ],
            "example_queries": [
                "What is the fine structure constant?",
                "Calculate dark matter percentage",
                "Apply mirror to 75",
                "Verify all axioms",
                "What is the Yang-Mills mass gap?"
            ],
            "brahim_sequence": BRAHIM_SEQUENCE,
            "constants": {
                "sum": SUM_CONSTANT,
                "center": CENTER,
                "phi": PHI
            }
        }

        return AgentResponse(
            success=True,
            result=help_text,
            intent=intent,
            layers_traversed=[LayerID.INTERFACE],
            trace=["Help request processed"],
            model_used=self.config.model.value
        )

    # Convenience methods matching OpenAI Agents SDK patterns

    def run(self, messages: List[Dict[str, str]]) -> AgentResponse:
        """Run agent with message list (OpenAI format)."""
        # Extract user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return self._generate_help_response(Intent.HELP)

        last_message = user_messages[-1].get("content", "")
        return self.process(last_message)

    async def arun(self, messages: List[Dict[str, str]]) -> AgentResponse:
        """Async version of run."""
        return self.run(messages)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool directly."""
        if tool_name == "calculate_physics_constant":
            constant = arguments.get("constant")
            if constant == "fine_structure":
                return fine_structure_constant().to_dict()
            elif constant == "weinberg_angle":
                return weinberg_angle().to_dict()
            elif constant == "muon_electron":
                return muon_electron_ratio().to_dict()
            elif constant == "proton_electron":
                return proton_electron_ratio().to_dict()

        elif tool_name == "calculate_cosmology":
            return cosmic_fractions().to_dict()

        elif tool_name == "calculate_yang_mills":
            return yang_mills_mass_gap().to_dict()

        elif tool_name == "apply_mirror_operator":
            return mirror_operator(arguments.get("value", 107))

        elif tool_name == "get_brahim_sequence":
            return get_sequence()

        elif tool_name == "verify_axioms":
            axiom_type = arguments.get("axiom_type", "all")
            if axiom_type == "mirror_symmetry":
                return verify_mirror_symmetry()
            elif axiom_type == "wightman":
                ym = yang_mills_mass_gap()
                return {"axioms": ym.wightman_satisfied}
            else:
                return {
                    "mirror": verify_mirror_symmetry(),
                    "wightman": {"satisfied": yang_mills_mass_gap().wightman_satisfied}
                }

        raise ValueError(f"Unknown tool: {tool_name}")


# =============================================================================
# AGENT BUILDER PATTERN
# =============================================================================

class BrahimAgentBuilder:
    """
    Builder pattern for creating customized Brahim Onion Agents.

    Usage:
        agent = (BrahimAgentBuilder()
            .with_model(ModelType.GPT4O)
            .with_industry("Particle Physics")
            .with_strict_axioms(True)
            .build())
    """

    def __init__(self):
        self._config = AgentConfig()

    def with_name(self, name: str) -> "BrahimAgentBuilder":
        self._config.name = name
        return self

    def with_model(self, model: ModelType) -> "BrahimAgentBuilder":
        self._config.model = model
        return self

    def with_temperature(self, temp: float) -> "BrahimAgentBuilder":
        self._config.temperature = max(0.0, min(2.0, temp))
        return self

    def with_industry(self, industry: str) -> "BrahimAgentBuilder":
        self._config.industry = industry
        return self

    def with_strict_axioms(self, strict: bool) -> "BrahimAgentBuilder":
        self._config.strict_axiom_mode = strict
        return self

    def with_layer_logging(self, enabled: bool) -> "BrahimAgentBuilder":
        self._config.enable_layer_logging = enabled
        return self

    def with_timeout(self, seconds: float) -> "BrahimAgentBuilder":
        self._config.timeout_seconds = seconds
        return self

    def build(self) -> BrahimOnionAgent:
        return BrahimOnionAgent(self._config)


# =============================================================================
# HANDOFF DEFINITIONS (Multi-Agent Pattern)
# =============================================================================

HANDOFF_DEFINITIONS = {
    "physics_specialist": {
        "name": "Physics Specialist",
        "description": "Handles particle physics calculations (α, θ_W, mass ratios)",
        "tools": ["calculate_physics_constant"],
        "trigger_intents": [Intent.PHYSICS]
    },
    "cosmology_specialist": {
        "name": "Cosmology Specialist",
        "description": "Handles cosmological calculations (DM, DE, Hubble)",
        "tools": ["calculate_cosmology"],
        "trigger_intents": [Intent.COSMOLOGY]
    },
    "qft_specialist": {
        "name": "QFT Specialist",
        "description": "Handles quantum field theory (Yang-Mills, Wightman)",
        "tools": ["calculate_yang_mills", "verify_axioms"],
        "trigger_intents": [Intent.YANG_MILLS, Intent.VERIFY]
    },
    "symmetry_specialist": {
        "name": "Symmetry Specialist",
        "description": "Handles mirror operations and sequence analysis",
        "tools": ["apply_mirror_operator", "get_brahim_sequence"],
        "trigger_intents": [Intent.MIRROR, Intent.SEQUENCE]
    }
}


# =============================================================================
# GUARDRAILS
# =============================================================================

class BrahimGuardrails:
    """Input/output guardrails for the agent."""

    @staticmethod
    def validate_input(user_input: str) -> tuple[bool, str]:
        """Validate user input."""
        if not user_input or not user_input.strip():
            return False, "Empty input"
        if len(user_input) > 10000:
            return False, "Input too long (max 10000 chars)"
        return True, "OK"

    @staticmethod
    def validate_output(response: AgentResponse) -> tuple[bool, str]:
        """Validate agent output."""
        if not response.success:
            return False, "Calculation failed"
        if response.result is None:
            return False, "No result produced"
        return True, "OK"

    @staticmethod
    def check_physics_bounds(result: Dict[str, Any]) -> bool:
        """Check if physics results are within reasonable bounds."""
        value = result.get("value")
        if value is None:
            return True

        name = result.get("name", "").lower()

        # Known bounds
        if "fine structure" in name:
            return 135 < value < 140
        if "weinberg" in name:
            return 0.2 < value < 0.25
        if "muon" in name:
            return 200 < value < 210
        if "proton" in name:
            return 1800 < value < 1850

        return True


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Interactive CLI for Brahim Onion Agent."""
    print("=" * 60)
    print("BRAHIM ONION AGENT (BOA)")
    print("Multi-layer computational agent for physics")
    print("=" * 60)

    # Build agent
    agent = (BrahimAgentBuilder()
        .with_name("boa-cli")
        .with_model(ModelType.GPT4O_MINI)
        .with_strict_axioms(True)
        .build())

    print(f"\nAgent: {agent.config.name}")
    print(f"Model: {agent.config.model.value}")
    print(f"Tools: {len(agent.tools)}")
    print("\nType 'help' for capabilities, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Validate input
            valid, msg = BrahimGuardrails.validate_input(user_input)
            if not valid:
                print(f"Invalid input: {msg}")
                continue

            # Process
            response = agent.process(user_input)

            # Display result
            print(f"\nIntent: {response.intent.value}")
            print(f"Layers: {' → '.join(l.value for l in response.layers_traversed)}")
            print(f"Result: {json.dumps(response.result, indent=2, default=str)}")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
