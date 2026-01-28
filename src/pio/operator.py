"""
PIO - Personal Intelligent Operator

Full async implementation with session management, intent routing,
and integration with GPIA reasoning and Moltbot channels.
"""

import asyncio
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Awaitable
from enum import Enum
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class SessionState(Enum):
    """Session lifecycle states."""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class IntentType(Enum):
    """Detected intent types."""
    QUERY = "query"           # Information request
    COMMAND = "command"       # Action request
    CONVERSATION = "conversation"  # General chat
    CREATIVE = "creative"     # Creative task
    ANALYSIS = "analysis"     # Analysis task
    UNKNOWN = "unknown"


@dataclass
class Message:
    """A message in the conversation."""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[IntentType] = None


@dataclass
class Session:
    """
    A conversation session with full state management.
    """
    id: str
    state: SessionState = SessionState.CREATED
    history: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    channel: Optional[str] = None
    user_id: Optional[str] = None

    def add_message(
        self,
        role: str,
        content: str,
        intent: Optional[IntentType] = None,
        metadata: Optional[Dict] = None,
    ) -> Message:
        """Add a message to the session history."""
        msg = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            intent=intent,
            metadata=metadata or {},
        )
        self.history.append(msg)
        self.updated_at = time.time()
        return msg

    def get_recent_history(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages."""
        return self.history[-n:]

    def format_history(self, n: int = 10) -> str:
        """Format recent history as a string for context."""
        messages = self.get_recent_history(n)
        lines = []
        for msg in messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)


class IntentDetector:
    """
    Detects user intent from messages.

    Uses pattern matching for fast classification.
    """

    PATTERNS = {
        IntentType.COMMAND: [
            "run", "execute", "do", "create", "delete", "update",
            "start", "stop", "send", "open", "close", "set",
        ],
        IntentType.QUERY: [
            "what", "how", "why", "when", "where", "who", "which",
            "explain", "tell me", "describe", "show",
        ],
        IntentType.CREATIVE: [
            "write", "generate", "compose", "design", "imagine",
            "invent", "brainstorm", "create a story",
        ],
        IntentType.ANALYSIS: [
            "analyze", "compare", "evaluate", "assess", "review",
            "examine", "investigate", "study",
        ],
    }

    def detect(self, text: str) -> IntentType:
        """Detect intent from text."""
        text_lower = text.lower()

        for intent, patterns in self.PATTERNS.items():
            if any(p in text_lower for p in patterns):
                return intent

        # Check if it's a question
        if text.strip().endswith("?"):
            return IntentType.QUERY

        return IntentType.CONVERSATION


# Type alias for middleware
Middleware = Callable[["PIOOperator", Session, str], Awaitable[Optional[str]]]


class PIOOperator:
    """
    Personal Intelligent Operator.

    Main interface for Sovereign PIO with full async support:
    - Session management
    - Intent detection and routing
    - Middleware pipeline
    - Integration with GPIA reasoning
    - Channel-agnostic processing
    """

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.intent_detector = IntentDetector()
        self._middleware: List[Middleware] = []
        self._reasoning_engine = None
        self._memory = None

    def use(self, middleware: Middleware) -> "PIOOperator":
        """Add middleware to the processing pipeline."""
        self._middleware.append(middleware)
        return self

    def set_reasoning_engine(self, engine) -> "PIOOperator":
        """Set the GPIA reasoning engine."""
        self._reasoning_engine = engine
        return self

    def set_memory(self, memory) -> "PIOOperator":
        """Set the GPIA memory system."""
        self._memory = memory
        return self

    def create_session(
        self,
        session_id: Optional[str] = None,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Session:
        """Create a new session."""
        sid = session_id or str(uuid.uuid4())
        session = Session(id=sid, channel=channel, user_id=user_id)
        session.state = SessionState.ACTIVE
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Session:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id, channel, user_id)
        return session

    def end_session(self, session_id: str) -> bool:
        """End a session."""
        session = self.get_session(session_id)
        if session:
            session.state = SessionState.ENDED
            return True
        return False

    async def process(
        self,
        session_id: str,
        user_input: str,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Process user input and return a response.

        Pipeline:
        1. Get/create session
        2. Detect intent
        3. Run middleware
        4. Route to appropriate handler
        5. Generate response
        6. Store in history

        Args:
            session_id: The session identifier
            user_input: User's input message
            channel: Optional channel identifier
            user_id: Optional user identifier

        Returns:
            Response string
        """
        # Get or create session
        session = self.get_or_create_session(session_id, channel, user_id)

        # Detect intent
        intent = self.intent_detector.detect(user_input)

        # Add user message to history
        session.add_message("user", user_input, intent=intent)

        # Run middleware pipeline
        response = None
        for middleware in self._middleware:
            result = await middleware(self, session, user_input)
            if result is not None:
                response = result
                break

        # If no middleware handled it, use default processing
        if response is None:
            response = await self._default_process(session, user_input, intent)

        # Add assistant response to history
        session.add_message("assistant", response)

        return response

    async def _default_process(
        self,
        session: Session,
        user_input: str,
        intent: IntentType,
    ) -> str:
        """Default processing when no middleware handles the request."""

        # If we have a reasoning engine, use it
        if self._reasoning_engine:
            # Map intent to task type
            task_map = {
                IntentType.QUERY: "reasoning",
                IntentType.COMMAND: "stability",
                IntentType.CREATIVE: "creativity",
                IntentType.ANALYSIS: "reasoning",
                IntentType.CONVERSATION: "harmony",
                IntentType.UNKNOWN: "reasoning",
            }
            task_type = task_map.get(intent, "reasoning")

            # Include conversation context
            context = session.format_history(5) if len(session.history) > 1 else None

            # Get memories if available
            if self._memory:
                memories = self._memory.search(user_input, top_k=3)
                if memories:
                    memory_context = "\n".join([
                        f"- {m.content}" for m, _ in memories
                    ])
                    context = f"Relevant memories:\n{memory_context}\n\n{context or ''}"

            result = await self._reasoning_engine.reason(
                query=user_input,
                task_type=task_type,
                context=context,
            )
            return result.response

        # Fallback response
        return f"[PIO] Received ({intent.value}): {user_input}"

    async def process_stream(
        self,
        session_id: str,
        user_input: str,
    ):
        """
        Process with streaming response.

        Yields tokens as they're generated.
        """
        session = self.get_or_create_session(session_id)
        intent = self.intent_detector.detect(user_input)
        session.add_message("user", user_input, intent=intent)

        if self._reasoning_engine:
            task_map = {
                IntentType.QUERY: "reasoning",
                IntentType.CREATIVE: "creativity",
                IntentType.ANALYSIS: "reasoning",
            }
            task_type = task_map.get(intent, "reasoning")

            full_response = []
            async for token in self._reasoning_engine.reason_stream(
                user_input, task_type
            ):
                full_response.append(token)
                yield token

            # Store complete response
            session.add_message("assistant", "".join(full_response))
        else:
            response = f"[PIO] Streaming not available for: {user_input}"
            session.add_message("assistant", response)
            yield response

    def stats(self) -> Dict[str, Any]:
        """Get operator statistics."""
        active = sum(1 for s in self.sessions.values() if s.state == SessionState.ACTIVE)
        total_messages = sum(len(s.history) for s in self.sessions.values())

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active,
            "total_messages": total_messages,
            "middleware_count": len(self._middleware),
            "has_reasoning": self._reasoning_engine is not None,
            "has_memory": self._memory is not None,
        }


# Pre-built middleware

async def logging_middleware(pio: PIOOperator, session: Session, input: str) -> Optional[str]:
    """Log all messages."""
    print(f"[LOG] Session {session.id}: {input[:50]}...")
    return None  # Continue to next middleware


async def memory_store_middleware(pio: PIOOperator, session: Session, input: str) -> Optional[str]:
    """Store important messages in memory."""
    if pio._memory and len(input) > 50:  # Store substantial messages
        key = f"msg_{session.id}_{int(time.time())}"
        pio._memory.store(
            key=key,
            content=input,
            metadata={"session_id": session.id, "type": "user_input"},
        )
    return None
