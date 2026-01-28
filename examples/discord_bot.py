"""
Sovereign PIO - Discord Bot Example

Run a Discord bot powered by Sovereign PIO.

Usage:
    export DISCORD_BOT_TOKEN="your-token-here"
    python discord_bot.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pio import PIOOperator
from gpia import Memory, ReasoningEngine, ModelConfig
from moltbot import Gateway, DiscordChannel, DiscordConfig


async def main():
    # Get token from environment
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("Error: DISCORD_BOT_TOKEN not set")
        print("Usage: export DISCORD_BOT_TOKEN='your-token' && python discord_bot.py")
        return

    print("Starting Sovereign PIO Discord Bot...")

    # Initialize components
    memory = Memory()
    pio = PIOOperator()
    pio.set_memory(memory)

    # Try to connect to Ollama
    try:
        reasoning = ReasoningEngine(ModelConfig.ollama())
        pio.set_reasoning_engine(reasoning)
        print("Connected to Ollama for reasoning")
    except Exception:
        reasoning = None
        print("Running without LLM")

    # Create Discord channel
    discord = DiscordChannel(DiscordConfig(
        bot_token=token,
        command_prefix="!pio ",  # Respond to !pio commands
        name="discord",
    ))

    # Wire PIO to Discord
    async def handle_message(message):
        return await pio.process(
            session_id=f"discord_{message.metadata.get('guild_id')}_{message.sender}",
            user_input=message.content,
            channel="discord",
            user_id=message.metadata.get("author_id"),
        )

    discord.on_message(handle_message)

    # Create gateway
    gateway = Gateway()
    gateway.register(discord)

    # Run
    print("Connecting to Discord...")
    await gateway.connect()
    print("Bot is running! Use '!pio <message>' to interact. Press Ctrl+C to stop.")

    try:
        await gateway.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await gateway.stop()
        if reasoning:
            await reasoning.close()


if __name__ == "__main__":
    asyncio.run(main())
