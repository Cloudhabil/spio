"""
Brahim Onion Agent - REST API Server

Lightweight HTTP server for mobile and web integration.
Can run standalone or embedded in APK via Chaquopy/Kivy.

Endpoints:
    POST /v1/chat/completions  - OpenAI-compatible chat
    POST /v1/tools/execute     - Direct tool execution
    GET  /v1/sequence          - Get Brahim sequence
    GET  /v1/health            - Health check

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from __future__ import annotations
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from urllib.parse import parse_qs, urlparse
import threading

# Import Brahim SDK
from ..agents_sdk import (
    BRAHIM_SEQUENCE,
    SUM_CONSTANT,
    CENTER,
    PHI,
    fine_structure_constant,
    weinberg_angle,
    muon_electron_ratio,
    proton_electron_ratio,
    cosmic_fractions,
    yang_mills_mass_gap,
    mirror_operator,
    get_sequence,
    verify_mirror_symmetry,
    BRAHIM_FUNCTIONS,
    execute_function,
)

from ..openai_agent import (
    BrahimOnionAgent,
    BrahimAgentBuilder,
    AgentConfig,
    ModelType,
    BRAHIM_AGENT_TOOLS,
)


# =============================================================================
# API RESPONSE MODELS
# =============================================================================

@dataclass
class APIResponse:
    """Standard API response format."""
    success: bool
    data: Any
    error: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str = "brahim-onion-agent"
    choices: List[Dict[str, Any]] = None
    usage: Dict[str, int] = None

    def __post_init__(self):
        if self.created == 0:
            self.created = int(time.time())
        if self.choices is None:
            self.choices = []
        if self.usage is None:
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class BrahimAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Brahim API."""

    # Class-level agent instance (shared)
    agent: BrahimOnionAgent = None

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _read_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        return {}

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._set_headers(204)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/v1/health":
            self._handle_health()
        elif path == "/v1/sequence":
            self._handle_sequence()
        elif path == "/v1/tools":
            self._handle_list_tools()
        elif path == "/v1/capabilities":
            self._handle_capabilities()
        elif path == "/":
            self._handle_root()
        else:
            self._handle_not_found()

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_body()

            if path == "/v1/chat/completions":
                self._handle_chat_completions(body)
            elif path == "/v1/tools/execute":
                self._handle_tool_execute(body)
            elif path == "/v1/physics":
                self._handle_physics(body)
            elif path == "/v1/cosmology":
                self._handle_cosmology()
            elif path == "/v1/yang-mills":
                self._handle_yang_mills()
            elif path == "/v1/mirror":
                self._handle_mirror(body)
            elif path == "/v1/verify":
                self._handle_verify(body)
            else:
                self._handle_not_found()

        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
        except Exception as e:
            self._send_error(500, str(e))

    # -------------------------------------------------------------------------
    # GET Handlers
    # -------------------------------------------------------------------------

    def _handle_root(self):
        """Root endpoint - API info."""
        info = {
            "name": "Brahim Onion Agent API",
            "version": "1.3.0",
            "description": "Multi-layer computational agent for physics calculations",
            "author": "Elias Oulad Brahim",
            "doi": "10.5281/zenodo.18356196",
            "endpoints": {
                "GET /v1/health": "Health check",
                "GET /v1/sequence": "Get Brahim sequence",
                "GET /v1/tools": "List available tools",
                "GET /v1/capabilities": "List capabilities by industry",
                "POST /v1/chat/completions": "OpenAI-compatible chat",
                "POST /v1/tools/execute": "Execute a tool directly",
                "POST /v1/physics": "Calculate physics constant",
                "POST /v1/cosmology": "Calculate cosmological fractions",
                "POST /v1/yang-mills": "Calculate Yang-Mills mass gap",
                "POST /v1/mirror": "Apply mirror operator",
                "POST /v1/verify": "Verify axioms",
            },
            "brahim_sequence": BRAHIM_SEQUENCE,
            "constants": {
                "sum": SUM_CONSTANT,
                "center": CENTER,
                "phi": PHI,
            }
        }
        self._send_json(info)

    def _handle_health(self):
        """Health check endpoint."""
        health = {
            "status": "healthy",
            "agent": "active" if self.agent else "not initialized",
            "timestamp": time.time(),
            "version": "1.3.0",
        }
        self._send_json(health)

    def _handle_sequence(self):
        """Get Brahim sequence endpoint."""
        self._send_json(get_sequence())

    def _handle_list_tools(self):
        """List available tools."""
        self._send_json({
            "tools": BRAHIM_AGENT_TOOLS,
            "count": len(BRAHIM_AGENT_TOOLS),
        })

    def _handle_capabilities(self):
        """List capabilities by industry."""
        from .config import APK_MANIFEST, INDUSTRY_PRESETS
        self._send_json({
            "capabilities": APK_MANIFEST["capabilities"],
            "industries": INDUSTRY_PRESETS,
        })

    def _handle_not_found(self):
        """404 handler."""
        self._send_error(404, f"Endpoint not found: {self.path}")

    # -------------------------------------------------------------------------
    # POST Handlers
    # -------------------------------------------------------------------------

    def _handle_chat_completions(self, body: Dict[str, Any]):
        """OpenAI-compatible chat completions endpoint."""
        messages = body.get("messages", [])

        if not messages:
            self._send_error(400, "No messages provided")
            return

        # Initialize agent if needed
        if not self.agent:
            self.agent = BrahimAgentBuilder().build()

        # Process through agent
        response = self.agent.run(messages)

        # Format as OpenAI response
        completion = ChatCompletionResponse(
            id=f"chatcmpl-brahim-{int(time.time())}",
            model=body.get("model", "brahim-onion-agent"),
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(response.result, default=str),
                },
                "finish_reason": "stop",
            }],
        )

        self._send_json(json.loads(completion.to_json()))

    def _handle_tool_execute(self, body: Dict[str, Any]):
        """Execute a tool directly."""
        tool_name = body.get("name") or body.get("tool")
        arguments = body.get("arguments", {})

        if not tool_name:
            self._send_error(400, "No tool name provided")
            return

        try:
            result = execute_function(tool_name, arguments)
            self._send_json({"result": result})
        except ValueError as e:
            self._send_error(400, str(e))

    def _handle_physics(self, body: Dict[str, Any]):
        """Calculate physics constant."""
        constant = body.get("constant", "fine_structure")

        result_map = {
            "fine_structure": fine_structure_constant,
            "weinberg_angle": weinberg_angle,
            "muon_electron": muon_electron_ratio,
            "proton_electron": proton_electron_ratio,
        }

        if constant not in result_map:
            self._send_error(400, f"Unknown constant: {constant}")
            return

        result = result_map[constant]()
        self._send_json(result.to_dict())

    def _handle_cosmology(self):
        """Calculate cosmological fractions."""
        result = cosmic_fractions()
        self._send_json(result.to_dict())

    def _handle_yang_mills(self):
        """Calculate Yang-Mills mass gap."""
        result = yang_mills_mass_gap()
        self._send_json(result.to_dict())

    def _handle_mirror(self, body: Dict[str, Any]):
        """Apply mirror operator."""
        value = body.get("value", 107)
        result = mirror_operator(value)
        self._send_json(result)

    def _handle_verify(self, body: Dict[str, Any]):
        """Verify axioms."""
        axiom_type = body.get("type", "all")

        if axiom_type == "mirror" or axiom_type == "mirror_symmetry":
            result = verify_mirror_symmetry()
        elif axiom_type == "wightman":
            ym = yang_mills_mass_gap()
            result = {"axioms": ym.wightman_satisfied, "all_satisfied": all(ym.wightman_satisfied)}
        else:
            result = {
                "mirror": verify_mirror_symmetry(),
                "wightman": {"satisfied": yang_mills_mass_gap().wightman_satisfied}
            }

        self._send_json(result)

    # -------------------------------------------------------------------------
    # Response Helpers
    # -------------------------------------------------------------------------

    def _send_json(self, data: Any, status: int = 200):
        """Send JSON response."""
        self._set_headers(status)
        response = json.dumps(data, default=str, indent=2)
        self.wfile.write(response.encode("utf-8"))

    def _send_error(self, status: int, message: str):
        """Send error response."""
        self._set_headers(status)
        error = {"error": message, "status": status}
        self.wfile.write(json.dumps(error).encode("utf-8"))

    def log_message(self, format: str, *args):
        """Override to customize logging."""
        print(f"[BOA API] {args[0]} {args[1]} {args[2]}")


# =============================================================================
# SERVER CLASS
# =============================================================================

class BrahimAPIServer:
    """
    Brahim Onion Agent API Server.

    Usage:
        server = BrahimAPIServer(host="0.0.0.0", port=8214)
        server.start()  # Blocking
        # or
        server.start_background()  # Non-blocking
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8214):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

        # Initialize shared agent
        BrahimAPIHandler.agent = BrahimAgentBuilder().build()

    def start(self):
        """Start server (blocking)."""
        self.server = HTTPServer((self.host, self.port), BrahimAPIHandler)
        print(f"Brahim Onion Agent API running on http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop")
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()

    def start_background(self) -> threading.Thread:
        """Start server in background thread."""
        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("Server stopped")


def create_app(host: str = "0.0.0.0", port: int = 8214) -> BrahimAPIServer:
    """Factory function to create API server."""
    return BrahimAPIServer(host=host, port=port)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Brahim Onion Agent API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8214, help="Port to listen on")
    args = parser.parse_args()

    server = BrahimAPIServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
