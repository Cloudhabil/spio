"""
Sovereign PIO CLI

Command-line interface for the Personal Intelligent Operator.

Subcommands:
    spio run              Boot runtime with terminal channel (echo mode)
    spio run --llm ollama Boot with Ollama LLM backend
    spio run --telegram T Boot with Telegram channel
    spio run --discord T  Boot with Discord channel
    spio status           Print unified status as JSON
"""

import argparse
import asyncio
import json
import sys

from . import __version__
from .constants import ALPHA, BETA, GAMMA, OMEGA, PHI


def print_banner():
    """Print the Sovereign PIO banner."""
    banner = """
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551                      SOVEREIGN PIO                             \u2551
\u2551         Personal Intelligent Operator \u00b7 Autonomous Edition     \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551  PIO \u2194\u2192 GPIA \u2194\u2192 ASIOS \u2194\u2192 MOLTBOT                              \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
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


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    """Boot and run the Sovereign Runtime."""
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(llm_provider=args.llm or "echo")

    if args.llm == "ollama":
        config.llm_model = args.model or "llama3.2"
        config.llm_host = args.host or "http://localhost:11434"
        if args.embed_model:
            config.embedding_model = args.embed_model
    elif args.llm == "openai":
        config.llm_model = args.model or "gpt-4o-mini"
        config.openai_api_key = args.api_key or ""

    if args.telegram:
        config.channel = "telegram"
        config.channel_token = args.telegram
    elif args.discord:
        config.channel = "discord"
        config.channel_token = args.discord
    else:
        config.channel = "terminal"

    print_banner()
    print(f"Version: {__version__}")
    print(f"LLM:     {config.llm_provider} ({config.llm_model})")
    print(f"Channel: {config.channel}")
    print()

    runtime = SovereignRuntime(config)
    runtime.boot()

    try:
        asyncio.run(runtime.run())
    except KeyboardInterrupt:
        asyncio.run(runtime.shutdown())

    return 0


def cmd_status(args) -> int:
    """Print unified runtime status as JSON."""
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(llm_provider="echo")
    runtime = SovereignRuntime(config)
    runtime.boot()

    status = runtime.status()
    print(json.dumps(status, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    sub = parser.add_subparsers(dest="command")

    # --- spio run ---
    run_parser = sub.add_parser("run", help="Boot and run the runtime")
    run_parser.add_argument(
        "--llm",
        choices=["echo", "ollama", "openai"],
        default=None,
        help="LLM provider (default: echo)",
    )
    run_parser.add_argument("--model", default=None, help="Model name")
    run_parser.add_argument("--host", default=None, help="Ollama host URL")
    run_parser.add_argument("--api-key", default=None, help="OpenAI API key")
    run_parser.add_argument("--embed-model", default=None, help="Ollama embedding model (default: nomic-embed-text)")
    run_parser.add_argument("--telegram", default=None, help="Telegram bot token")
    run_parser.add_argument("--discord", default=None, help="Discord bot token")

    # --- spio status ---
    sub.add_parser("status", help="Print unified runtime status")

    # --- legacy flags (no subcommand) ---
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

    # Dispatch subcommands
    if args.command == "run":
        return cmd_run(args)
    if args.command == "status":
        return cmd_status(args)

    # Legacy flag mode
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
    print("Run 'spio run' to start the runtime.")
    print("Run 'spio status' to see system status.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
