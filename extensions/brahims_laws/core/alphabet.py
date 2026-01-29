"""
Brahim's Laws Compressed JSON Alphabet.

A compact notation system for efficient representation of elliptic curve
data and analysis results. Each symbol maps to a semantic category.

Alphabet Structure:
    C_* : Curve identifiers and invariants
    L_* : Law computations (1-6)
    R_* : Reynolds/Regime related
    S_* : Sha (Tate-Shafarevich) related
    M_* : ML/Model predictions
    E_* : Exponents and constants
    V_* : Validation/Verification
    A_* : Audit/Hash related
    P_* : Pipeline/Processing metadata

Usage:
    from brahims_laws.core.alphabet import Alphabet, compress, expand

    # Compress full result to compact form
    compact = compress(full_result)

    # Expand compact form back to full
    full = expand(compact)

Author: Elias Oulad Brahim
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class Category(Enum):
    """Alphabet categories."""
    CURVE = "C"       # Curve data
    LAW = "L"         # Law computations
    REYNOLDS = "R"    # Reynolds/Regime
    SHA = "S"         # Sha values
    MODEL = "M"       # ML predictions
    EXPONENT = "E"    # Exponents/Constants
    VALIDATION = "V"  # Verification
    AUDIT = "A"       # Audit trail
    PIPELINE = "P"    # Processing metadata


# =============================================================================
# ALPHABET DEFINITION
# =============================================================================

ALPHABET = {
    # -------------------------------------------------------------------------
    # C_* : Curve Identifiers and Invariants
    # -------------------------------------------------------------------------
    "C_l": "label",                    # Cremona/LMFDB label
    "C_N": "conductor",                # Conductor N
    "C_r": "rank",                     # Algebraic rank
    "C_O": "real_period",              # Real period Omega
    "C_T": "tamagawa_product",         # Tamagawa product
    "C_t": "im_tau",                   # Im(tau)
    "C_R": "regulator",                # Regulator
    "C_a": "a_invariants",             # Weierstrass coefficients
    "C_to": "torsion_order",           # Torsion order
    "C_src": "source",                 # Data source

    # -------------------------------------------------------------------------
    # L_* : Law Computations (Laws 1-6)
    # -------------------------------------------------------------------------
    "L_1m": "law1_sha_median",         # Law 1: Sha from Im(tau)^(2/3)
    "L_1o": "law1_sha_omega",          # Law 1: Sha from Omega^(-4/3)
    "L_1e": "law1_error",              # Law 1: Relative error

    "L_2": "law2_reynolds",            # Law 2: Reynolds number

    "L_3r": "law3_regime",             # Law 3: Regime classification
    "L_3l": "law3_rey_lower",          # Law 3: Lower threshold (10)
    "L_3u": "law3_rey_upper",          # Law 3: Upper threshold (30)

    "L_4m": "law4_sha_max",            # Law 4: Sha_max ~ Rey^(5/12)
    "L_4e": "law4_error",              # Law 4: Relative error

    "L_5v": "law5_variance",           # Law 5: log(Sha) variance
    "L_5p": "law5_p_exponent",         # Law 5: Fitted p-exponent

    "L_6v": "law6_verified",           # Law 6: Consistency verified
    "L_6e": "law6_error",              # Law 6: Consistency error

    # -------------------------------------------------------------------------
    # R_* : Reynolds and Regime
    # -------------------------------------------------------------------------
    "R_n": "reynolds_number",          # Reynolds number
    "R_g": "regime",                   # Regime: LAM/TRA/TUR
    "R_lam": "is_laminar",             # Is laminar (Rey < 10)
    "R_tra": "is_transition",          # Is transition (10-30)
    "R_tur": "is_turbulent",           # Is turbulent (Rey > 30)
    "R_m": "mean_reynolds",            # Mean Reynolds (batch)
    "R_s": "std_reynolds",             # Std Reynolds (batch)

    # -------------------------------------------------------------------------
    # S_* : Sha (Tate-Shafarevich Group)
    # -------------------------------------------------------------------------
    "S_a": "sha_analytic",             # Analytic Sha (ground truth)
    "S_p": "sha_predicted",            # Predicted Sha
    "S_m": "sha_max",                  # Sha max bound
    "S_1": "sha_is_trivial",           # Sha == 1
    "S_n": "sha_nontrivial_prob",      # P(Sha > 1)
    "S_e": "sha_error",                # Prediction error

    # -------------------------------------------------------------------------
    # M_* : ML Model Predictions
    # -------------------------------------------------------------------------
    "M_sp": "ml_sha_pred",             # ML Sha prediction
    "M_rp": "ml_regime_pred",          # ML regime prediction
    "M_rpr": "ml_regime_probs",        # ML regime probabilities
    "M_emb": "ml_embedding",           # Learned embedding
    "M_unc": "ml_uncertainty",         # Prediction uncertainty
    "M_r2": "ml_r2_score",             # Model R2 score
    "M_f1": "ml_f1_score",             # Model F1 score

    # -------------------------------------------------------------------------
    # E_* : Exponents and Constants
    # -------------------------------------------------------------------------
    "E_a": "alpha",                    # Alpha exponent (2/3)
    "E_g": "gamma",                    # Gamma exponent (5/12)
    "E_d": "delta",                    # Delta exponent (-1/4)
    "E_C": "calibration_const",        # Calibration constant C
    "E_at": "alpha_theoretical",       # Theoretical alpha
    "E_gt": "gamma_theoretical",       # Theoretical gamma
    "E_dt": "delta_theoretical",       # Theoretical delta
    "E_ae": "alpha_error",             # Alpha error from theory
    "E_ge": "gamma_error",             # Gamma error from theory
    "E_de": "delta_error",             # Delta error from theory

    # -------------------------------------------------------------------------
    # V_* : Validation and Verification
    # -------------------------------------------------------------------------
    "V_c": "consistency_verified",     # Law 6 consistency check
    "V_ce": "consistency_error",       # Consistency relation error
    "V_bl": "brahim_laws_valid",       # Overall Brahim's Laws valid
    "V_bsd": "bsd_compatible",         # BSD conjecture compatible
    "V_ph": "physics_loss",            # PINN physics loss
    "V_dl": "data_loss",               # PINN data loss

    # -------------------------------------------------------------------------
    # A_* : Audit Trail
    # -------------------------------------------------------------------------
    "A_h": "vnand_hash",               # VNAND hash
    "A_mh": "master_hash",             # Batch master hash
    "A_ts": "timestamp",               # Timestamp
    "A_v": "version",                  # Schema version

    # -------------------------------------------------------------------------
    # P_* : Pipeline Metadata
    # -------------------------------------------------------------------------
    "P_n": "n_curves",                 # Number of curves
    "P_t": "n_turns",                  # Number of turns
    "P_c": "current_turn",             # Current turn
    "P_cum": "cumulative_curves",      # Cumulative processed
    "P_ms": "processing_time_ms",      # Processing time (ms)
    "P_tp": "throughput",              # Throughput (curves/sec)
}

# Reverse mapping for expansion
ALPHABET_REVERSE = {v: k for k, v in ALPHABET.items()}

# Regime abbreviations
REGIME_COMPACT = {
    "laminar": "LAM",
    "transition": "TRA",
    "turbulent": "TUR",
    "LAMINAR": "LAM",
    "TRANSITION": "TRA",
    "TURBULENT": "TUR",
}

REGIME_EXPAND = {v: k.lower() for k, v in REGIME_COMPACT.items() if k.islower()}


# =============================================================================
# COMPRESSION / EXPANSION FUNCTIONS
# =============================================================================

def compress(data: Dict[str, Any], include_nulls: bool = False) -> Dict[str, Any]:
    """
    Compress a full result dictionary to compact alphabet notation.

    Args:
        data: Full result dictionary with verbose keys
        include_nulls: Include null/None values in output

    Returns:
        Compressed dictionary with alphabet keys
    """
    result = {}

    for full_key, value in data.items():
        # Get compact key
        compact_key = ALPHABET_REVERSE.get(full_key, full_key)

        # Skip nulls if not wanted
        if value is None and not include_nulls:
            continue

        # Compress nested dicts
        if isinstance(value, dict):
            value = compress(value, include_nulls)

        # Compress lists of dicts
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            value = [compress(item, include_nulls) for item in value]

        # Compress regime values
        elif full_key in ['regime', 'law3_regime', 'ml_regime_pred']:
            if isinstance(value, str):
                value = REGIME_COMPACT.get(value, value)

        # Round floats for compactness
        elif isinstance(value, float):
            if abs(value) < 0.0001:
                value = round(value, 8)
            elif abs(value) < 1:
                value = round(value, 6)
            else:
                value = round(value, 4)

        result[compact_key] = value

    return result


def expand(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand a compressed dictionary back to full verbose keys.

    Args:
        data: Compressed dictionary with alphabet keys

    Returns:
        Expanded dictionary with full keys
    """
    result = {}

    for compact_key, value in data.items():
        # Get full key
        full_key = ALPHABET.get(compact_key, compact_key)

        # Expand nested dicts
        if isinstance(value, dict):
            value = expand(value)

        # Expand lists of dicts
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            value = [expand(item) for item in value]

        # Expand regime values
        elif compact_key in ['R_g', 'L_3r', 'M_rp']:
            if isinstance(value, str):
                value = REGIME_EXPAND.get(value, value)

        result[full_key] = value

    return result


def compress_json(data: Dict[str, Any], indent: Optional[int] = None) -> str:
    """Compress and serialize to JSON string."""
    compressed = compress(data)
    return json.dumps(compressed, indent=indent, default=str)


def expand_json(json_str: str) -> Dict[str, Any]:
    """Parse JSON and expand to full keys."""
    data = json.loads(json_str)
    return expand(data)


# =============================================================================
# SCHEMA DOCUMENTATION
# =============================================================================

@dataclass
class AlphabetSchema:
    """Schema documentation for the alphabet."""

    @staticmethod
    def get_schema() -> Dict[str, Dict[str, str]]:
        """Get full schema grouped by category."""
        schema = {
            "C": {"name": "Curve", "desc": "Curve identifiers and invariants", "keys": {}},
            "L": {"name": "Law", "desc": "Law 1-6 computations", "keys": {}},
            "R": {"name": "Reynolds", "desc": "Reynolds number and regime", "keys": {}},
            "S": {"name": "Sha", "desc": "Tate-Shafarevich group", "keys": {}},
            "M": {"name": "Model", "desc": "ML predictions", "keys": {}},
            "E": {"name": "Exponent", "desc": "Scaling exponents", "keys": {}},
            "V": {"name": "Validation", "desc": "Verification results", "keys": {}},
            "A": {"name": "Audit", "desc": "Audit trail", "keys": {}},
            "P": {"name": "Pipeline", "desc": "Processing metadata", "keys": {}},
        }

        for compact, full in ALPHABET.items():
            category = compact.split("_")[0]
            if category in schema:
                schema[category]["keys"][compact] = full

        return schema

    @staticmethod
    def print_schema():
        """Print human-readable schema."""
        schema = AlphabetSchema.get_schema()

        print("=" * 70)
        print("BRAHIM'S LAWS - COMPRESSED JSON ALPHABET")
        print("=" * 70)

        for cat_code, cat_data in schema.items():
            print(f"\n{cat_code}_* : {cat_data['name']} - {cat_data['desc']}")
            print("-" * 50)
            for compact, full in cat_data['keys'].items():
                print(f"  {compact:12s} -> {full}")

        print("\n" + "=" * 70)
        print("REGIME ABBREVIATIONS")
        print("-" * 30)
        for full, compact in REGIME_COMPACT.items():
            if full.islower():
                print(f"  {compact} -> {full}")

    @staticmethod
    def to_markdown() -> str:
        """Generate markdown documentation."""
        schema = AlphabetSchema.get_schema()

        md = ["# Brahim's Laws Compressed JSON Alphabet\n"]
        md.append("A compact notation system for efficient data representation.\n")

        for cat_code, cat_data in schema.items():
            md.append(f"\n## {cat_data['name']} (`{cat_code}_*`)\n")
            md.append(f"{cat_data['desc']}\n")
            md.append("\n| Compact | Full Key | Description |")
            md.append("|---------|----------|-------------|")

            for compact, full in cat_data['keys'].items():
                desc = full.replace("_", " ").title()
                md.append(f"| `{compact}` | `{full}` | {desc} |")

        md.append("\n## Regime Abbreviations\n")
        md.append("| Code | Meaning |")
        md.append("|------|---------|")
        for full, compact in REGIME_COMPACT.items():
            if full.islower():
                md.append(f"| `{compact}` | {full} |")

        return "\n".join(md)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_compressed_curve():
    """Example of a compressed curve result."""
    return {
        "C_l": "11a1",
        "C_N": 11,
        "C_r": 0,
        "C_O": 1.2692,
        "C_T": 1,
        "C_t": 0.9533,
        "S_a": 1,
        "L_1m": 1.02,
        "L_1e": 0.02,
        "L_2": 8.67,
        "L_3r": "LAM",
        "L_4m": 2.1,
        "L_6v": True,
        "E_a": 0.6667,
        "E_g": 0.4167,
        "E_d": -0.25,
        "A_h": "a3f8c2d1...",
        "A_ts": "2026-01-22T22:33:10"
    }


def example_compressed_batch():
    """Example of compressed batch/pipeline result."""
    return {
        "P_n": 100,
        "P_t": 10,
        "P_ms": 24920,
        "P_tp": 4.0,
        "R_m": 29352.84,
        "R_s": 56645.84,
        "S_1": 56,
        "S_n": 44,
        "V_c": True,
        "E_a": 0.6195,
        "E_g": 0.4128,
        "E_d": -0.2064,
        "E_ae": 0.0471,
        "E_ge": 0.0039,
        "E_de": 0.0436,
        "A_mh": "4957597ef954...",
        "A_ts": "2026-01-22T22:33:10"
    }


# CLI for testing
if __name__ == "__main__":
    AlphabetSchema.print_schema()

    print("\n\nEXAMPLE: Compressed Curve")
    print("-" * 40)
    print(json.dumps(example_compressed_curve(), indent=2))

    print("\n\nEXAMPLE: Expanded Curve")
    print("-" * 40)
    print(json.dumps(expand(example_compressed_curve()), indent=2))
