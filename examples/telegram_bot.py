"""
Sovereign PIO - Telegram Bot Example

Run a Telegram bot powered by Sovereign PIO.

Usage:
    export TELEGRAM_BOT_TOKEN="your-token-here"
    python telegram_bot.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pio import PIOOperator
from gpia import Memory, ReasoningEngine, ModelConfig
from moltbot import Gateway, TelegramChannel, TelegramConfig


async def main():
    # Get token from environment
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        print("Usage: export TELEGRAM_BOT_TOKEN='your-token' && python telegram_bot.py")
        return

    print("Starting Sovereign PIO Telegram Bot...")

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

    # Create Telegram channel
    telegram = TelegramChannel(TelegramConfig(
        bot_token=token,
        name="telegram",
    ))

    # Wire PIO to Telegram
    async def handle_message(message):
        return await pio.process(
            session_id=f"telegram_{message.sender}",
            user_input=message.content,
            channel="telegram",
            user_id=message.metadata.get("user_id"),
        )

    telegram.on_message(handle_message)

    # Create gateway
    gateway = Gateway()
    gateway.register(telegram)

    # Run
    print("Connecting to Telegram...")
    await gateway.connect()
    print("Bot is running! Press Ctrl+C to stop.")

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
