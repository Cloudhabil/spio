"""
Agents Extension - Multi-Agent Orchestration System

10 Components for autonomous agent coordination:

1. AgentMemory        - SQLite-based episodic/semantic memory
2. LessonManager      - Lesson exchange between teacher/student agents
3. AlphaAgent         - Autonomous learning agent
4. ProfessorAgent     - Autonomous teaching agent
5. SessionOrchestrator - Coordinates learning sessions
6. ModelRouter        - Routes queries to appropriate LLMs
7. BudgetAllocator    - ML-based resource allocation
8. NeuronicRouter     - PASS-integrated model routing
9. AgentRegistry      - Central registry of agents
10. TeachingTopics    - Curriculum for agent learning

Mathematical Foundation:
- PHI-based timing (cycle intervals)
- BETA-based memory decay
- Brahim sequence for priority levels

Reference: GPIA Multi-Agent Architecture
"""

from .agents_core import (
    # Constants
    PHI,
    BETA,
    GAMMA,
    SESSION_DURATION_DEFAULT,
    CYCLE_INTERVAL_DEFAULT,
    LEARNING_CYCLES_DEFAULT,
    BRAHIM_SEQUENCE,

    # Enums
    MemoryType,
    TaskUrgency,
    AgentTier,
    ModelRole,

    # 1. AgentMemory
    AgentMemory,

    # 2. LessonManager
    LessonManager,

    # 3. AlphaAgent
    AlphaAgent,

    # 4. ProfessorAgent
    ProfessorAgent,
    BaseAgent,

    # 5. SessionOrchestrator
    SessionOrchestrator,

    # 6. ModelRouter
    ModelRouter,
    ModelConfig,
    MODELS,
    TASK_ROUTING,

    # 7. BudgetAllocator
    BudgetAllocator,
    AllocationRequest,
    DecisionFactors,
    DecisionWeights,

    # 8. NeuronicRouter
    NeuronicRouter,

    # 9. AgentRegistry
    AgentRegistry,

    # 10. TeachingTopics
    TeachingTopic,
    TeachingTopicsCatalog,
    TEACHING_TOPICS,

    # Factories
    create_learning_session,
    create_model_router,
    create_neuronic_router,
    create_agent_registry,
)

__all__ = [
    # Constants
    "PHI",
    "BETA",
    "GAMMA",
    "SESSION_DURATION_DEFAULT",
    "CYCLE_INTERVAL_DEFAULT",
    "LEARNING_CYCLES_DEFAULT",
    "BRAHIM_SEQUENCE",

    # Enums
    "MemoryType",
    "TaskUrgency",
    "AgentTier",
    "ModelRole",

    # 1. AgentMemory
    "AgentMemory",

    # 2. LessonManager
    "LessonManager",

    # 3. AlphaAgent
    "AlphaAgent",

    # 4. ProfessorAgent
    "ProfessorAgent",
    "BaseAgent",

    # 5. SessionOrchestrator
    "SessionOrchestrator",

    # 6. ModelRouter
    "ModelRouter",
    "ModelConfig",
    "MODELS",
    "TASK_ROUTING",

    # 7. BudgetAllocator
    "BudgetAllocator",
    "AllocationRequest",
    "DecisionFactors",
    "DecisionWeights",

    # 8. NeuronicRouter
    "NeuronicRouter",

    # 9. AgentRegistry
    "AgentRegistry",

    # 10. TeachingTopics
    "TeachingTopic",
    "TeachingTopicsCatalog",
    "TEACHING_TOPICS",

    # Factories
    "create_learning_session",
    "create_model_router",
    "create_neuronic_router",
    "create_agent_registry",
]
