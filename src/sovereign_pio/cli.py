"""
Sovereign PIO CLI

Command-line interface for the Personal Intelligent Operator.
"""

import argparse
import sys

from . import __version__
from .constants import PHI, ALPHA, OMEGA, BETA, GAMMA


def print_banner():
    """Print the Sovereign PIO banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                      SOVEREIGN PIO                             ║
║         Personal Intelligent Operator · Autonomous Edition     ║
╠═══════════════════════════════════════════════════════════════╣
║  PIO ←→ GPIA ←→ ASIOS ←→ MOLTBOT                              ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_constants():
    """Print Brahim's Calculator constants."""
    print("\nBrahim's Calculator Constants:")
    print(f"  PHI   = {PHI:.16f}")
    print(f"  ALPHA = {ALPHA:.16f} (Creation)")
    print(f"  OMEGA = {OMEGA:.16f} (Return)")
    print(f"  BETA  = {BETA:.16f} (Security)")
    print(f"  GAMMA = {GAMMA:.16f} (Damping)")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="spio",
        description="Sovereign PIO - Personal Intelligent Operator",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Sovereign PIO v{__version__}",
    )
    parser.add_argument(
        "--constants", "-c",
        action="store_true",
        help="Display Brahim's Calculator constants",
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Display system information",
    )

    args = parser.parse_args()

    print_banner()
    print(f"Version: {__version__}")

    if args.constants:
        print_constants()
        return 0

    if args.info:
        print("\nArchitecture Layers:")
        print("  1. PIO     - Personal Intelligent Operator (Interface)")
        print("  2. GPIA    - Intelligence & Reasoning")
        print("  3. ASIOS   - Operating System Runtime")
        print("  4. MOLTBOT - Multi-Channel Gateway")
        print_constants()
        return 0

    # Default: show help
    print("\nRun 'spio --help' for available commands.")
    print("Run 'spio --info' for system information.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
