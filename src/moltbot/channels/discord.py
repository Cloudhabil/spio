"""
Discord Channel

Discord bot integration using Discord's HTTP API and Gateway.
"""

import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx

from .base import Channel, ChannelConfig, ChannelType, Message


@dataclass
class DiscordConfig(ChannelConfig):
    """Discord-specific configuration."""
    bot_token: str = ""
    application_id: str = ""
    allowed_guilds: List[str] = None  # None = allow all
    allowed_channels: List[str] = None
    command_prefix: str = "!"

    def __post_init__(self):
        self.channel_type = ChannelType.DISCORD
        if self.allowed_guilds is None:
            self.allowed_guilds = []
        if self.allowed_channels is None:
            self.allowed_channels = []


class DiscordChannel(Channel):
    """
    Discord bot channel.

    Features:
    - HTTP API for sending messages
    - Gateway WebSocket for receiving events
    - Guild/channel allowlists
    - Embed support
    """

    API_BASE = "https://discord.com/api/v10"
    GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self.discord_config: DiscordConfig = config
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bot {config.bot_token}"},
        )
        self._ws = None
        self._heartbeat_interval = 0
        self._running = False
        self._session_id = None
        self._sequence = None

    async def _api_call(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a Discord API call."""
        url = f"{self.API_BASE}{endpoint}"

        if method == "GET":
            response = await self._client.get(url, params=kwargs)
        elif method == "POST":
            response = await self._client.post(url, json=kwargs)
        elif method == "PATCH":
            response = await self._client.patch(url, json=kwargs)
        elif method == "DELETE":
            response = await self._client.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 204:
            return {}

        response.raise_for_status()
        return response.json()

    async def connect(self) -> bool:
        """Connect to Discord."""
        try:
            # Verify token
            user = await self._api_call("GET", "/users/@me")
            print(f"Connected as {user.get('username')}#{user.get('discriminator')}")
            self.connected = True
            return True
        except Exception as e:
            print(f"Discord connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Discord."""
        self._running = False
        if self._ws:
            await self._ws.close()
        await self._client.aclose()
        self.connected = False
        return True

    async def send(
        self,
        recipient: str,  # channel_id
        content: str,
        embed: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """Send a message to a channel."""
        try:
            payload = {"content": content}

            if embed:
                payload["embeds"] = [embed]

            if "reply_to" in kwargs:
                payload["message_reference"] = {"message_id": kwargs["reply_to"]}

            await self._api_call("POST", f"/channels/{recipient}/messages", **payload)
            return True

        except Exception as e:
            print(f"Discord send failed: {e}")
            return False

    def _is_allowed(self, guild_id: str, channel_id: str) -> bool:
        """Check if guild/channel is allowed."""
        # Check guild allowlist
        if self.discord_config.allowed_guilds:
            if guild_id not in self.discord_config.allowed_guilds:
                return False

        # Check channel allowlist
        if self.discord_config.allowed_channels:
            if channel_id not in self.discord_config.allowed_channels:
                return False

        return True

    async def _handle_gateway_message(self, data: Dict[str, Any]) -> None:
        """Handle a gateway message."""
        op = data.get("op")
        event_type = data.get("t")
        payload = data.get("d", {})

        # Update sequence
        if data.get("s"):
            self._sequence = data["s"]

        # Handle opcodes
        if op == 10:  # Hello
            self._heartbeat_interval = payload["heartbeat_interval"] / 1000
            # Send identify
            await self._send_gateway({
                "op": 2,
                "d": {
                    "token": self.discord_config.bot_token,
                    "intents": 513,  # GUILDS + GUILD_MESSAGES
                    "properties": {
                        "os": "linux",
                        "browser": "sovereign-pio",
                        "device": "sovereign-pio",
                    },
                },
            })

        elif op == 11:  # Heartbeat ACK
            pass

        elif op == 0:  # Dispatch
            if event_type == "READY":
                self._session_id = payload.get("session_id")
                print("Discord gateway ready")

            elif event_type == "MESSAGE_CREATE":
                await self._handle_message(payload)

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle an incoming message."""
        # Ignore bot messages
        if data.get("author", {}).get("bot"):
            return

        channel_id = data.get("channel_id", "")
        guild_id = data.get("guild_id", "")
        content = data.get("content", "")

        # Check allowlist
        if not self._is_allowed(guild_id, channel_id):
            return

        # Check for command prefix
        if self.discord_config.command_prefix:
            if not content.startswith(self.discord_config.command_prefix):
                return
            content = content[len(self.discord_config.command_prefix):].strip()

        # Create message
        message = Message(
            id=data.get("id", str(uuid.uuid4())),
            channel=ChannelType.DISCORD,
            sender=channel_id,
            content=content,
            metadata={
                "guild_id": guild_id,
                "author_id": data.get("author", {}).get("id"),
                "author_name": data.get("author", {}).get("username"),
                "message_id": data.get("id"),
            },
        )

        # Dispatch
        response = await self._dispatch(message)

        # Send response
        if response:
            await self.send(channel_id, response, reply_to=data.get("id"))

    async def _send_gateway(self, data: Dict[str, Any]) -> None:
        """Send data to gateway."""
        if self._ws:
            await self._ws.send(json.dumps(data))

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to gateway."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            await self._send_gateway({"op": 1, "d": self._sequence})

    async def listen(self) -> None:
        """Connect to gateway and listen for events."""
        import websockets

        self._running = True

        while self._running:
            try:
                async with websockets.connect(self.GATEWAY_URL) as ws:
                    self._ws = ws

                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                    try:
                        async for message in ws:
                            data = json.loads(message)
                            await self._handle_gateway_message(data)
                    finally:
                        heartbeat_task.cancel()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Discord gateway error: {e}")
                await asyncio.sleep(5)

    async def create_embed(
        self,
        title: str,
        description: str,
        color: int = 0x5865F2,
        fields: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a Discord embed."""
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }

        if fields:
            embed["fields"] = fields

        return embed
