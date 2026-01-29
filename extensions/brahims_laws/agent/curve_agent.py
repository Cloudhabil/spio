"""
Elliptic Curve Analysis Agent using OpenAI Agents SDK.

An AI agent that can analyze elliptic curves against Brahim's Laws,
answer questions about BSD conjecture, and explain mathematical concepts.

Author: Elias Oulad Brahim
"""

import json
from typing import Optional
from fractions import Fraction

# Try to import agents SDK (optional dependency)
try:
    from agents import Agent, Runner, function_tool
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    # Create dummy decorators
    def function_tool(f):
        return f

# Import Brahim's Laws components
from ..core.brahim_laws import BrahimLawsEngine
from ..core.constants import CONSTANTS
from ..models.curve_data import EllipticCurveData, Regime
from ..data.lmfdb_client import LMFDBClient
from ..audit.vnand_hasher import VNANDHasher


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@function_tool
def analyze_curve_by_label(label: str) -> str:
    """
    Analyze an elliptic curve by its Cremona or LMFDB label.

    Args:
        label: The curve label (e.g., '11a1', '37.a1', '389a1')

    Returns:
        JSON string with complete Brahim's Laws analysis
    """
    client = LMFDBClient()
    curve = client.fetch_by_label(label)

    if curve is None:
        return json.dumps({
            "error": f"Could not find curve with label '{label}'",
            "suggestion": "Try using LMFDB format (e.g., '11.a1') or Cremona format (e.g., '11a1')"
        })

    engine = BrahimLawsEngine()
    result = engine.analyze(curve)

    hasher = VNANDHasher()
    hasher.hash_and_attach(result)

    return json.dumps(result.to_dict(), indent=2, default=str)


@function_tool
def analyze_curve_by_coefficients(a1: int, a2: int, a3: int, a4: int, a6: int) -> str:
    """
    Analyze an elliptic curve from its Weierstrass coefficients.

    The curve equation is: y^2 + a1*xy + a3*y = x^3 + a2*x^2 + a4*x + a6

    Args:
        a1: First coefficient
        a2: Second coefficient
        a3: Third coefficient
        a4: Fourth coefficient
        a6: Sixth coefficient (note: a5 is not used in standard form)

    Returns:
        JSON string with analysis (note: some BSD data may be estimated)
    """
    curve = EllipticCurveData.from_coefficients(a1, a2, a3, a4, a6)

    # Set placeholder values (full computation requires SageMath)
    curve.conductor = 11  # Placeholder
    curve.real_period = 1.0
    curve.tamagawa_product = 1
    curve.im_tau = 1.0
    curve.sha_analytic = 1

    engine = BrahimLawsEngine()
    result = engine.analyze(curve)

    return json.dumps({
        "curve": curve.weierstrass_equation,
        "note": "Full BSD invariants require SageMath or LMFDB lookup",
        "analysis": result.to_dict()
    }, indent=2, default=str)


@function_tool
def verify_consistency_relation() -> str:
    """
    Verify Brahim's Law 6: The consistency relation 2/3 = 5/12 + 1/4.

    This fundamental relation connects:
    - alpha (2/3): Geometric exponent from Law 1 (Brahim Conjecture)
    - gamma (5/12): Dynamic exponent from Law 4 (Reynolds scaling)
    - delta (1/4): Cascade exponent from Law 5 (prime variance decay)

    Returns:
        JSON string with verification results
    """
    lhs = Fraction(2, 3)
    rhs = Fraction(5, 12) + Fraction(1, 4)

    return json.dumps({
        "law": "Law 6: Consistency Relation",
        "relation": "alpha = gamma + |delta|",
        "components": {
            "alpha (geometric)": {"fraction": "2/3", "decimal": float(Fraction(2, 3))},
            "gamma (dynamic)": {"fraction": "5/12", "decimal": float(Fraction(5, 12))},
            "|delta| (cascade)": {"fraction": "1/4", "decimal": float(Fraction(1, 4))}
        },
        "lhs": {"expression": "2/3", "value": float(lhs)},
        "rhs": {"expression": "5/12 + 1/4", "value": float(rhs)},
        "difference": abs(float(lhs) - float(rhs)),
        "verified": lhs == rhs,
        "interpretation": "The exact equality suggests Laws 1, 4, and 5 are manifestations of a single underlying structure."
    }, indent=2)


@function_tool
def get_brahim_constants() -> str:
    """
    Get all Brahim's Laws constants and their mathematical meanings.

    Returns:
        JSON string with all constants and their interpretations
    """
    return json.dumps({
        "constants": {
            "ALPHA_IMTAU": {
                "value": CONSTANTS.ALPHA_IMTAU,
                "fraction": "2/3",
                "meaning": "Geometric exponent: Sha ~ Im(tau)^(2/3)",
                "law": "Law 1 (Brahim Conjecture)"
            },
            "BETA_OMEGA": {
                "value": CONSTANTS.BETA_OMEGA,
                "fraction": "-4/3",
                "meaning": "Period exponent: Sha ~ Omega^(-4/3)",
                "law": "Law 1 (alternate form)"
            },
            "GAMMA_REY": {
                "value": CONSTANTS.GAMMA_REY,
                "fraction": "5/12",
                "meaning": "Reynolds scaling: Sha_max ~ Rey^(5/12)",
                "law": "Law 4 (Dynamic Scaling)"
            },
            "DELTA_CASCADE": {
                "value": CONSTANTS.DELTA_CASCADE,
                "fraction": "-1/4",
                "meaning": "Cascade decay: Var(log Sha | p) ~ p^(-1/4)",
                "law": "Law 5 (Cascade Law)"
            },
            "REY_C_LOWER": {
                "value": CONSTANTS.REY_C_LOWER,
                "meaning": "Lower critical Reynolds number (laminar/transition boundary)",
                "law": "Law 3 (Phase Transition)"
            },
            "REY_C_UPPER": {
                "value": CONSTANTS.REY_C_UPPER,
                "meaning": "Upper critical Reynolds number (transition/turbulent boundary)",
                "law": "Law 3 (Phase Transition)"
            }
        },
        "key_relation": "2/3 = 5/12 + 1/4 (Law 6: Consistency)"
    }, indent=2)


@function_tool
def explain_reynolds_regimes() -> str:
    """
    Explain the three Reynolds regimes and their implications for Sha.

    Returns:
        JSON string explaining laminar, transition, and turbulent regimes
    """
    return json.dumps({
        "law": "Law 3: Phase Transition",
        "definition": "Reynolds Number Rey = N / (Tam * Omega)",
        "regimes": {
            "LAMINAR": {
                "condition": f"Rey < {CONSTANTS.REY_C_LOWER}",
                "sha_behavior": "Sha = 1 almost always",
                "physical_analogy": "Smooth, predictable flow",
                "probability_sha_gt_1": "Very low (<5%)"
            },
            "TRANSITION": {
                "condition": f"{CONSTANTS.REY_C_LOWER} <= Rey <= {CONSTANTS.REY_C_UPPER}",
                "sha_behavior": "Mixed: some curves have Sha > 1",
                "physical_analogy": "Intermittent turbulent bursts",
                "probability_sha_gt_1": "Moderate (5-20%)"
            },
            "TURBULENT": {
                "condition": f"Rey > {CONSTANTS.REY_C_UPPER}",
                "sha_behavior": "Sha > 1 becomes common",
                "physical_analogy": "Chaotic, unpredictable flow",
                "probability_sha_gt_1": "Higher (>20%)"
            }
        },
        "insight": "The transition mirrors fluid dynamics: small Reynolds = laminar flow, large Reynolds = turbulence."
    }, indent=2)


@function_tool
def search_curves_by_rank(rank: int, limit: int = 10) -> str:
    """
    Search for elliptic curves with a specific rank from LMFDB.

    Args:
        rank: The algebraic rank (0, 1, 2, etc.)
        limit: Maximum number of curves to return (default 10)

    Returns:
        JSON string with list of curves and brief analysis
    """
    client = LMFDBClient()
    curves = client.fetch_batch(rank=rank, limit=limit)

    if not curves:
        return json.dumps({
            "error": f"No curves found with rank {rank}",
            "note": "LMFDB connection may be unavailable"
        })

    engine = BrahimLawsEngine()
    results = []

    for curve in curves[:limit]:
        analysis = engine.analyze(curve)
        results.append({
            "label": curve.label,
            "conductor": curve.conductor,
            "rank": curve.rank,
            "reynolds": round(analysis.reynolds_number, 2),
            "regime": str(analysis.regime),
            "sha_predicted": round(analysis.sha_median_predicted, 4)
        })

    return json.dumps({
        "query": {"rank": rank, "limit": limit},
        "found": len(results),
        "curves": results
    }, indent=2)


@function_tool
def explain_bsd_conjecture() -> str:
    """
    Explain the Birch and Swinnerton-Dyer (BSD) Conjecture in accessible terms.

    Returns:
        JSON string with explanation of BSD conjecture
    """
    return json.dumps({
        "name": "Birch and Swinnerton-Dyer Conjecture",
        "status": "Millennium Prize Problem (unsolved, $1M prize)",
        "statement": {
            "informal": "The rank of an elliptic curve equals the order of vanishing of its L-function at s=1",
            "formula": "rank(E) = ord_{s=1} L(E, s)"
        },
        "key_components": {
            "L-function": "A complex analytic function encoding arithmetic data of the curve",
            "Rank": "Number of independent rational points of infinite order",
            "Sha": "Tate-Shafarevich group - measures failure of local-global principle"
        },
        "bsd_formula": "L^(r)(E,1) / r! = (Omega * |Sha| * prod(c_p) * Reg) / |E_tors|^2",
        "brahim_connection": "Brahim's Laws provide statistical constraints on Sha that complement BSD predictions",
        "relevance": "Understanding Sha distribution helps verify BSD predictions empirically"
    }, indent=2)


# =============================================================================
# AGENT DEFINITION
# =============================================================================

SYSTEM_INSTRUCTIONS = """You are an expert mathematician specializing in elliptic curves,
the Birch and Swinnerton-Dyer (BSD) conjecture, and Brahim's Laws.

Your capabilities:
1. Analyze elliptic curves against all 6 Brahim's Laws
2. Explain mathematical concepts in accessible terms
3. Search and retrieve curve data from LMFDB
4. Verify mathematical relations and computations

Brahim's Six Laws:
- Law 1 (Brahim Conjecture): Sha_median ~ Im(tau)^(2/3) ~ Omega^(-4/3)
- Law 2 (Reynolds Number): Rey = N / (Tam * Omega)
- Law 3 (Phase Transition): Critical Reynolds ~ 10-30
- Law 4 (Dynamic Scaling): Sha_max ~ Rey^(5/12)
- Law 5 (Cascade Law): Var(log Sha | p) ~ p^(-1/4)
- Law 6 (Consistency): 2/3 = 5/12 + 1/4 exactly

When analyzing curves:
- Always explain the significance of results
- Connect findings to the broader BSD conjecture context
- Highlight which regime (laminar/transition/turbulent) the curve falls into
- Note any unusual or interesting patterns

Be precise with mathematics but accessible in explanations. Use the tools
available to perform actual computations rather than approximating."""


TOOLS = [
    analyze_curve_by_label,
    analyze_curve_by_coefficients,
    verify_consistency_relation,
    get_brahim_constants,
    explain_reynolds_regimes,
    search_curves_by_rank,
    explain_bsd_conjecture,
]


def create_agent():
    """Create the Brahim's Laws analysis agent."""
    if not HAS_AGENTS:
        raise ImportError("OpenAI Agents SDK not installed. Install with: pip install openai-agents")

    return Agent(
        name="Brahim's Laws Analyst",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=TOOLS,
        model="gpt-4o"
    )


# Alias for backwards compatibility
CurveAnalysisAgent = create_agent


def run_agent(query: str, model: str = "gpt-4o") -> str:
    """
    Run the agent with a query and return the response.

    Args:
        query: User's question or request
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Agent's response as string
    """
    if not HAS_AGENTS:
        raise ImportError("OpenAI Agents SDK not installed")

    agent = Agent(
        name="Brahim's Laws Analyst",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=TOOLS,
        model=model
    )

    result = Runner.run_sync(agent, query)
    return result.final_output


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def interactive_session():
    """Run an interactive session with the agent."""
    if not HAS_AGENTS:
        print("Error: OpenAI Agents SDK not installed")
        print("Install with: pip install openai-agents")
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    if HAS_RICH:
        console = Console()
        agent = create_agent()

        console.print(Panel(
            "[bold blue]Brahim's Laws Analysis Agent[/bold blue]\n\n"
            "I can help you analyze elliptic curves, explain BSD conjecture,\n"
            "and explore Brahim's Laws. Ask me anything!\n\n"
            "Type 'quit' or 'exit' to end the session.",
            title="Welcome",
            border_style="blue"
        ))

        while True:
            try:
                query = console.input("\n[bold green]You:[/bold green] ")

                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[dim]Goodbye![/dim]")
                    break

                if not query.strip():
                    continue

                console.print("[dim]Thinking...[/dim]")

                result = Runner.run_sync(agent, query)

                console.print("\n[bold blue]Agent:[/bold blue]")
                try:
                    console.print(Markdown(result.final_output))
                except:
                    console.print(result.final_output)

            except KeyboardInterrupt:
                console.print("\n[dim]Session interrupted. Goodbye![/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    else:
        print("=" * 60)
        print("Brahim's Laws Analysis Agent")
        print("=" * 60)
        print("Note: Install 'rich' for better formatting")
        print("Type 'quit' or 'exit' to end.")
        print()

        agent = create_agent()

        while True:
            try:
                query = input("You: ")

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not query.strip():
                    continue

                print("Thinking...")
                result = Runner.run_sync(agent, query)
                print(f"\nAgent: {result.final_output}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    interactive_session()
