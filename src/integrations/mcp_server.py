"""
Model Context Protocol (MCP) Server

Exposes Sovereign PIO memory databases to agentic monitoring tools.
Implements JSON-RPC 2.0 protocol over stdio.

Tools:
- query_memories: Execute read-only SQL against memory database
- get_memory_schema: Get database schema
- get_wavelength_stats: Get 12-wavelength pipeline statistics
- get_ignorance_map: Get current ignorance cartography
"""

import sqlite3
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# =============================================================================
# MCP SERVER
# =============================================================================

class MCPServer:
    """
    Model Context Protocol server for Sovereign PIO.

    Provides read-only access to:
    - Memory database (embeddings, conversations)
    - Wavelength statistics
    - Ignorance cartography
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._default_db_path()
        self.name = "sovereign-pio-mcp"
        self.version = "1.618.1"

    def _default_db_path(self) -> str:
        """Get default database path."""
        return str(Path(__file__).parent.parent.parent / "data" / "memories.db")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    # =========================================================================
    # TOOL DEFINITIONS
    # =========================================================================

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        return [
            {
                "name": "query_memories",
                "description": "Execute a read-only SQL query against the Sovereign PIO memory database.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute (SELECT only)."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_memory_schema",
                "description": "Get the schema of the memory database.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_wavelength_stats",
                "description": "Get 12-wavelength cognitive pipeline statistics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_hardware_stats",
                "description": "Get measured silicon hardware statistics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name."""
        if name == "query_memories":
            return self._query_memories(arguments.get("query", ""))
        elif name == "get_memory_schema":
            return self._get_schema()
        elif name == "get_wavelength_stats":
            return self._get_wavelength_stats()
        elif name == "get_hardware_stats":
            return self._get_hardware_stats()
        else:
            raise ValueError(f"Unknown tool: {name}")

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _query_memories(self, query: str) -> Dict[str, Any]:
        """Execute read-only SQL query."""
        # Security: only allow SELECT
        if not query.strip().upper().startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed for safety."}

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return {"results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    def _get_schema(self) -> Dict[str, Any]:
        """Get database schema."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
            tables = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return {"tables": tables}
        except Exception as e:
            return {"error": str(e)}

    def _get_wavelength_stats(self) -> Dict[str, Any]:
        """Get wavelength pipeline statistics."""
        try:
            from core.wavelengths import WavelengthGate, PRIME_TARGET_DENSITY
            gate = WavelengthGate()
            stats = gate.get_stats()
            stats["genesis_constant"] = PRIME_TARGET_DENSITY
            return stats
        except ImportError:
            return {
                "error": "Wavelength module not available",
                "genesis_constant": 0.00221975,  # 2/901
            }

    def _get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware statistics."""
        try:
            from core.measured_silicon import hardware_stats
            return hardware_stats()
        except ImportError:
            return {"error": "Hardware module not available"}

    # =========================================================================
    # JSON-RPC 2.0 PROTOCOL
    # =========================================================================

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a JSON-RPC 2.0 request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                result = {"content": [{"type": "text", "text": json.dumps(self.call_tool(tool_name, arguments))}]}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }

            return {"jsonrpc": "2.0", "id": req_id, "result": result}

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)}
            }

    def run_stdio(self):
        """
        Run JSON-RPC 2.0 server over stdio.

        Reads JSON requests from stdin, writes responses to stdout.
        """
        print(f"Starting {self.name} v{self.version}", file=sys.stderr)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }
                print(json.dumps(error_response), flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Sovereign PIO MCP Server")
    parser.add_argument("--db", help="Path to memory database")
    args = parser.parse_args()

    server = MCPServer(db_path=args.db)
    server.run_stdio()


if __name__ == "__main__":
    main()
