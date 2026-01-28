"""
Moltbot Gateway

Central hub managing multiple communication channels.
Routes messages between PIO and external platforms.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field

from .channels.base import Channel, ChannelType, Message


# Handler type
GatewayHandler = Callable[[Message], Awaitable[Optional[str]]]


@dataclass
class GatewayConfig:
    """Gateway configuration."""
    name: str = "sovereign-pio"
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0


class Gateway:
    """
    Moltbot Gateway - Multi-Channel Message Hub.

    Manages multiple channels and routes messages through
    a unified handler pipeline.

    Features:
    - Register multiple channels
    - Unified message handling
    - Automatic reconnection
    - Channel health monitoring
    - Broadcast messaging
    """

    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.channels: Dict[str, Channel] = {}
        self._handlers: List[GatewayHandler] = []
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def register(self, channel: Channel, name: Optional[str] = None) -> "Gateway":
        """
        Register a channel with the gateway.

        Args:
            channel: The channel instance
            name: Optional name override

        Returns:
            self for chaining
        """
        channel_name = name or channel.name
        self.channels[channel_name] = channel

        # Wire up message handler
        channel.on_message(self._handle_message)

        return self

    def on_message(self, handler: GatewayHandler) -> "Gateway":
        """
        Register a message handler.

        Handlers are called in order until one returns a response.
        """
        self._handlers.append(handler)
        return self

    async def _handle_message(self, message: Message) -> Optional[str]:
        """Route message through handler pipeline."""
        for handler in self._handlers:
            try:
                response = await handler(message)
                if response is not None:
                    return response
            except Exception as e:
                print(f"Handler error: {e}")
        return None

    async def connect(self, channel_name: Optional[str] = None) -> Dict[str, bool]:
        """
        Connect channels.

        Args:
            channel_name: Specific channel to connect, or None for all

        Returns:
            Dict of channel names to connection status
        """
        results = {}

        if channel_name:
            channels = {channel_name: self.channels.get(channel_name)}
        else:
            channels = self.channels

        for name, channel in channels.items():
            if channel is None:
                results[name] = False
                continue

            try:
                results[name] = await channel.connect()
            except Exception as e:
                print(f"Failed to connect {name}: {e}")
                results[name] = False

        return results

    async def disconnect(self, channel_name: Optional[str] = None) -> Dict[str, bool]:
        """Disconnect channels."""
        results = {}

        if channel_name:
            channels = {channel_name: self.channels.get(channel_name)}
        else:
            channels = self.channels

        for name, channel in channels.items():
            if channel is None:
                continue
            try:
                results[name] = await channel.disconnect()
            except Exception as e:
                print(f"Failed to disconnect {name}: {e}")
                results[name] = False

        return results

    async def send(
        self,
        channel_name: str,
        recipient: str,
        content: str,
        **kwargs,
    ) -> bool:
        """Send a message through a specific channel."""
        channel = self.channels.get(channel_name)
        if not channel or not channel.connected:
            return False

        return await channel.send(recipient, content, **kwargs)

    async def broadcast(
        self,
        content: str,
        channels: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple channels.

        Args:
            content: Message content
            channels: List of channel names (None = all connected)
            **kwargs: Additional arguments for send

        Returns:
            Dict of channel names to success status
        """
        results = {}

        target_channels = channels or list(self.channels.keys())

        for name in target_channels:
            channel = self.channels.get(name)
            if not channel or not channel.connected:
                results[name] = False
                continue

            # Broadcast needs a recipient - use '*' as wildcard
            results[name] = await channel.send("*", content, **kwargs)

        return results

    async def start(self) -> None:
        """
        Start all channels and listen for messages.

        This runs until stop() is called.
        """
        self._running = True

        # Connect all channels
        await self.connect()

        # Start listeners
        for name, channel in self.channels.items():
            if channel.connected:
                task = asyncio.create_task(
                    self._channel_listener(name, channel)
                )
                self._tasks.append(task)

        # Wait for all tasks
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _channel_listener(self, name: str, channel: Channel) -> None:
        """Run listener for a single channel with auto-reconnect."""
        while self._running:
            try:
                await channel.listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Channel {name} error: {e}")

                if self.config.auto_reconnect:
                    await asyncio.sleep(self.config.reconnect_delay)
                    try:
                        await channel.connect()
                    except Exception:
                        pass
                else:
                    break

    async def stop(self) -> None:
        """Stop the gateway and all channels."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Disconnect all channels
        await self.disconnect()

        self._tasks.clear()

    def status(self) -> Dict[str, Any]:
        """Get gateway status."""
        return {
            "name": self.config.name,
            "running": self._running,
            "channels": {
                name: channel.status()
                for name, channel in self.channels.items()
            },
            "handler_count": len(self._handlers),
        }

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name."""
        return self.channels.get(name)

    def list_channels(self) -> List[str]:
        """List all registered channel names."""
        return list(self.channels.keys())

    def connected_channels(self) -> List[str]:
        """List connected channel names."""
        return [
            name for name, channel in self.channels.items()
            if channel.connected
        ]
