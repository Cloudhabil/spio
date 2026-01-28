"""
PIO - Personal Intelligent Operator

The interface layer of Sovereign PIO.
Handles user interaction, session management, and intent routing.
"""

__all__ = ["PIOOperator", "Session"]


class Session:
    """Represents a user session with PIO."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.context = {}

    def add_message(self, role: str, content: str):
        """Add a message to the session history."""
        self.history.append({"role": role, "content": content})

    def get_context(self) -> dict:
        """Get the current session context."""
        return self.context.copy()


class PIOOperator:
    """
    Personal Intelligent Operator.

    The main interface for interacting with Sovereign PIO.
    Routes user intents through GPIA for reasoning and ASIOS for execution.
    """

    def __init__(self):
        self.sessions = {}
        self.active_session = None

    def create_session(self, session_id: str) -> Session:
        """Create a new session."""
        session = Session(session_id)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get an existing session."""
        return self.sessions.get(session_id)

    async def process(self, session_id: str, user_input: str) -> str:
        """
        Process user input and return a response.

        Args:
            session_id: The session identifier
            user_input: User's input message

        Returns:
            Response from the system
        """
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)

        session.add_message("user", user_input)

        # TODO: Route through GPIA for reasoning
        # TODO: Execute through ASIOS runtime
        # TODO: Return response via appropriate channel (Moltbot)

        response = f"[PIO] Received: {user_input}"
        session.add_message("assistant", response)

        return response
