"""
Gateway Extension - API Server and Substrate Gateway

Ported from:
- CLI-main/src/server/substrate_gateway.py
- CLI-main/src/server/main.py
- CLI-main/src/server/models.py
- CLI-main/src/server/config.py
- CLI-main/src/server/security.py
- CLI-main/src/server/db.py
- CLI-main/src/server/redis_bus.py

Implements:
- GatewayConfig: Pydantic settings for API/Redis/DB
- GatewaySecurity: Token validation for REST and WebSocket
- DatabasePool: Async PostgreSQL connection pooling
- RedisBus: PubSub bridge for real-time messaging
- SubstrateGateway: ASIOS Level 2 AC inference endpoint
- GatewayServer: Full FastAPI backend with node management
"""

from .gateway_core import (
    # Config
    GatewayConfig, get_gateway_config,

    # Models
    SeedBody, NodeConfig, NodeStatus, QueryRequest, QueryResponse,

    # Security
    GatewaySecurity, validate_token, validate_ws_token,

    # Database
    DatabasePool, get_db_pool, init_db, close_db, ping_db,

    # Redis
    RedisBus, get_redis_bus, publish, subscribe, psubscribe,

    # Substrate Gateway
    SubstrateGateway, create_substrate_gateway,

    # Server
    GatewayServer, create_gateway_server,

    # Factory
    create_gateway_app,
)

__all__ = [
    # Config
    "GatewayConfig", "get_gateway_config",
    # Models
    "SeedBody", "NodeConfig", "NodeStatus", "QueryRequest", "QueryResponse",
    # Security
    "GatewaySecurity", "validate_token", "validate_ws_token",
    # Database
    "DatabasePool", "get_db_pool", "init_db", "close_db", "ping_db",
    # Redis
    "RedisBus", "get_redis_bus", "publish", "subscribe", "psubscribe",
    # Substrate Gateway
    "SubstrateGateway", "create_substrate_gateway",
    # Server
    "GatewayServer", "create_gateway_server",
    # Factory
    "create_gateway_app",
]
