"""
Terminal Channel

Interactive terminal/CLI channel for local testing and development.
"""

import asyncio
import uuid
import sys
from typing import Optional

from .base import Channel, ChannelConfig, ChannelType, Message


class TerminalChannel(Channel):
    """
    Terminal-based channel for CLI interaction.

    Provides an interactive prompt for testing Sovereign PIO locally.
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        if config is None:
            config = ChannelConfig(
                channel_type=ChannelType.TERMINAL,
                name="terminal",
            )
        super().__init__(config)
        self._running = False
        self._prompt = config.settings.get("prompt", "You> ")
        self._bot_name = config.settings.get("bot_name", "PIO")

    async def connect(self) -> bool:
        """Connect (no-op for terminal)."""
        self.connected = True
        print(f"\n{'='*50}")
        print("  Sovereign PIO - Terminal Channel")
        print("  Type 'exit' or 'quit' to end session")
        print(f"{'='*50}\n")
        return True

    async def disconnect(self) -> bool:
        """Disconnect."""
        self._running = False
        self.connected = False
        print("\nGoodbye!")
        return True

    async def send(self, recipient: str, content: str, **kwargs) -> bool:
        """Send (print) a message."""
        print(f"{self._bot_name}> {content}")
        return True

    async def listen(self) -> None:
        """
        Start interactive terminal session.

        Reads from stdin and dispatches to handlers.
        """
        self._running = True

        while self._running:
            try:
                # Read input
                if sys.platform == 'win32':
                    # Windows doesn't support asyncio stdin well
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input(self._prompt)
                    )
                else:
                    print(self._prompt, end='', flush=True)
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    line = line.strip()

                # Check for exit
                if line.lower() in ('exit', 'quit', '/quit', '/exit'):
                    await self.disconnect()
                    break

                if not line:
                    continue

                # Create message
                message = Message(
                    id=str(uuid.uuid4()),
                    channel=ChannelType.TERMINAL,
                    sender="user",
                    content=line,
                )

                # Dispatch to handlers
                response = await self._dispatch(message)

                # Send response if any
                if response:
                    await self.send("user", response)

            except EOFError:
                await self.disconnect()
                break
            except KeyboardInterrupt:
                await self.disconnect()
                break
            except Exception as e:
                print(f"Error: {e}")

    async def run_once(self, user_input: str) -> Optional[str]:
        """
        Process a single input without interactive loop.

        Useful for testing and scripting.
        """
        message = Message(
            id=str(uuid.uuid4()),
            channel=ChannelType.TERMINAL,
            sender="user",
            content=user_input,
        )
        return await self._dispatch(message)
