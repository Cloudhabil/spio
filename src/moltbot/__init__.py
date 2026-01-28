"""
Moltbot - Multi-Channel Messaging Gateway

The communication layer of Sovereign PIO.
Handles multi-platform messaging through various channels.

Supported Channels:
- WhatsApp
- Telegram
- Discord
- Signal
- Slack
- iMessage
- Matrix
- Line
- Teams
- Web
- Terminal
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

__all__ = ["Channel", "ChannelType", "Message", "Gateway"]


class ChannelType(Enum):
    """Supported channel types."""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SIGNAL = "signal"
    SLACK = "slack"
    IMESSAGE = "imessage"
    MATRIX = "matrix"
    LINE = "line"
    TEAMS = "teams"
    WEB = "web"
    TERMINAL = "terminal"


@dataclass
class Message:
    """Represents a message in the system."""
    channel: ChannelType
    sender: str
    content: str
    timestamp: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Channel(Protocol):
    """Protocol for channel implementations."""

    async def send(self, recipient: str, content: str) -> bool:
        """Send a message through the channel."""
        ...

    async def receive(self) -> Message | None:
        """Receive a message from the channel."""
        ...

    async def connect(self) -> bool:
        """Connect to the channel."""
        ...

    async def disconnect(self) -> bool:
        """Disconnect from the channel."""
        ...


class Gateway:
    """
    Moltbot Gateway.

    Central hub for managing multiple communication channels.
    Routes messages between PIO and external platforms.
    """

    def __init__(self):
        self.channels: dict[ChannelType, Channel] = {}
        self.connected: set[ChannelType] = set()

    def register_channel(self, channel_type: ChannelType, channel: Channel):
        """Register a channel with the gateway."""
        self.channels[channel_type] = channel

    async def connect_channel(self, channel_type: ChannelType) -> bool:
        """Connect a specific channel."""
        if channel_type not in self.channels:
            return False

        channel = self.channels[channel_type]
        if await channel.connect():
            self.connected.add(channel_type)
            return True
        return False

    async def connect_all(self) -> dict[ChannelType, bool]:
        """Connect all registered channels."""
        results = {}
        for channel_type in self.channels:
            results[channel_type] = await self.connect_channel(channel_type)
        return results

    async def send(self, channel_type: ChannelType, recipient: str, content: str) -> bool:
        """Send a message through a specific channel."""
        if channel_type not in self.connected:
            return False

        channel = self.channels[channel_type]
        return await channel.send(recipient, content)

    async def broadcast(self, content: str, channels: list[ChannelType] = None) -> dict[ChannelType, bool]:
        """Broadcast a message to multiple channels."""
        targets = channels or list(self.connected)
        results = {}
        for channel_type in targets:
            if channel_type in self.connected:
                results[channel_type] = await self.send(channel_type, "*", content)
        return results

    def get_status(self) -> dict:
        """Get gateway status."""
        return {
            "registered": list(self.channels.keys()),
            "connected": list(self.connected),
        }
