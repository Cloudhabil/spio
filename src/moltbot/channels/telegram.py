"""
Telegram Channel

Full Telegram Bot API integration for messaging.
"""

import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import httpx

from .base import Channel, ChannelConfig, ChannelType, Message


@dataclass
class TelegramConfig(ChannelConfig):
    """Telegram-specific configuration."""
    bot_token: str = ""
    allowed_users: List[str] = None  # None = allow all
    polling_interval: float = 1.0

    def __post_init__(self):
        self.channel_type = ChannelType.TELEGRAM
        if self.allowed_users is None:
            self.allowed_users = []


class TelegramChannel(Channel):
    """
    Telegram Bot channel.

    Features:
    - Long polling for updates
    - Message sending with Markdown support
    - User allowlist
    - Reply and inline keyboard support
    """

    API_BASE = "https://api.telegram.org/bot"

    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self.telegram_config: TelegramConfig = config
        self._client = httpx.AsyncClient(timeout=60.0)
        self._last_update_id = 0
        self._running = False

    @property
    def _api_url(self) -> str:
        return f"{self.API_BASE}{self.telegram_config.bot_token}"

    async def _api_call(self, method: str, **params) -> Dict[str, Any]:
        """Make a Telegram API call."""
        url = f"{self._api_url}/{method}"
        response = await self._client.post(url, json=params)
        response.raise_for_status()
        data = response.json()

        if not data.get("ok"):
            raise Exception(f"Telegram API error: {data.get('description')}")

        return data.get("result", {})

    async def connect(self) -> bool:
        """Connect and verify bot token."""
        try:
            me = await self._api_call("getMe")
            print(f"Connected as @{me.get('username')}")
            self.connected = True
            return True
        except Exception as e:
            print(f"Telegram connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect."""
        self._running = False
        await self._client.aclose()
        self.connected = False
        return True

    async def send(
        self,
        recipient: str,
        content: str,
        parse_mode: str = "Markdown",
        reply_to: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Send a message to a chat."""
        try:
            params = {
                "chat_id": recipient,
                "text": content,
                "parse_mode": parse_mode,
            }

            if reply_to:
                params["reply_to_message_id"] = reply_to

            # Add optional keyboard
            if "keyboard" in kwargs:
                params["reply_markup"] = kwargs["keyboard"]

            await self._api_call("sendMessage", **params)
            return True

        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def _is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed."""
        if not self.telegram_config.allowed_users:
            return True  # Allow all if not configured
        return str(user_id) in self.telegram_config.allowed_users

    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Process a single update from Telegram."""
        # Handle message updates
        msg_data = update.get("message") or update.get("edited_message")
        if not msg_data:
            return

        # Extract info
        chat_id = str(msg_data["chat"]["id"])
        user_id = str(msg_data["from"]["id"])
        username = msg_data["from"].get("username", user_id)
        text = msg_data.get("text", "")

        # Check allowlist
        if not self._is_allowed(user_id):
            return

        # Create message
        message = Message(
            id=str(msg_data["message_id"]),
            channel=ChannelType.TELEGRAM,
            sender=chat_id,
            content=text,
            metadata={
                "user_id": user_id,
                "username": username,
                "chat_type": msg_data["chat"]["type"],
                "message_id": msg_data["message_id"],
            },
        )

        # Dispatch
        response = await self._dispatch(message)

        # Send response
        if response:
            await self.send(
                chat_id,
                response,
                reply_to=msg_data["message_id"],
            )

    async def listen(self) -> None:
        """Start long polling for updates."""
        self._running = True

        while self._running and self.connected:
            try:
                # Get updates
                updates = await self._api_call(
                    "getUpdates",
                    offset=self._last_update_id + 1,
                    timeout=30,
                )

                for update in updates:
                    self._last_update_id = update["update_id"]
                    await self._process_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Telegram polling error: {e}")
                await asyncio.sleep(5)

            await asyncio.sleep(self.telegram_config.polling_interval)

    async def send_photo(
        self,
        chat_id: str,
        photo_url: str,
        caption: Optional[str] = None,
    ) -> bool:
        """Send a photo."""
        try:
            params = {"chat_id": chat_id, "photo": photo_url}
            if caption:
                params["caption"] = caption
            await self._api_call("sendPhoto", **params)
            return True
        except Exception as e:
            print(f"Telegram send photo failed: {e}")
            return False

    async def send_document(
        self,
        chat_id: str,
        document_url: str,
        caption: Optional[str] = None,
    ) -> bool:
        """Send a document."""
        try:
            params = {"chat_id": chat_id, "document": document_url}
            if caption:
                params["caption"] = caption
            await self._api_call("sendDocument", **params)
            return True
        except Exception as e:
            print(f"Telegram send document failed: {e}")
            return False
