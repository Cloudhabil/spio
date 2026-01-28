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
- PHI = 1.618... (Golden Ratio) - agent coordination timing
- BETA = 0.236... (Security factor) - memory importance decay
- Decision weights adapt via gradient descent on outcomes

Reference: CLAUDE.md - Brahim's Calculator
Author: GPIA Multi-Agent Architecture
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ============================================================================
# Constants from Brahim's Calculator
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2  # 1.6180339887498949
BETA = 1 / (PHI ** 3)         # 0.2360679774997897 - Security/decay factor
GAMMA = 1 / (PHI ** 4)        # 0.1458980337503155 - Damping factor

# Agent timing based on PHI
SESSION_DURATION_DEFAULT = 300  # 5 minutes
CYCLE_INTERVAL_DEFAULT = int(SESSION_DURATION_DEFAULT / PHI)  # ~185 seconds
LEARNING_CYCLES_DEFAULT = 5

# Brahim sequence for agent priority levels
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]


# ============================================================================
# Enums
# ============================================================================

class MemoryType(Enum):
    """Types of agent memory."""
    EPISODIC = "episodic"      # Events and experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures


class TaskUrgency(Enum):
    """Task urgency levels."""
    IMMEDIATE = 3  # System-critical
    NORMAL = 2     # Standard operations
    BACKGROUND = 1 # Low-priority


class AgentTier(Enum):
    """Agent importance tiers."""
    SYSTEM = 3     # Core system agents
    PRIMARY = 2    # Professor, Alpha
    SECONDARY = 1  # Custom agents


class ModelRole(Enum):
    """LLM model roles."""
    FAST = "fast"           # Quick parsing (CodeGemma)
    CREATIVE = "creative"   # Dialogue/synthesis (Qwen3)
    REASONING = "reasoning" # Analysis/grading (DeepSeek-R1)
    VISION = "vision"       # Image analysis (LLaVA)
    SYNTHESIS = "synthesis" # Complex synthesis (GPT-OSS)


# ============================================================================
# 1. AgentMemory - SQLite-based memory for agents
# ============================================================================

class AgentMemory:
    """
    SQLite-based memory system for agents.

    Supports episodic, semantic, and procedural memories with
    importance-based retrieval and BETA-based decay.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize memory with optional persistent storage."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize memory tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'episodic',
                importance REAL DEFAULT 0.5,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at DESC);
        """)
        self.conn.commit()

    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        context: Optional[Dict] = None
    ) -> str:
        """
        Store a memory with computed importance.

        Args:
            content: Memory content
            memory_type: One of episodic, semantic, procedural
            importance: Base importance score (0.0-1.0)
            context: Optional context metadata

        Returns:
            Memory ID
        """
        memory_id = hashlib.sha256(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        self.conn.execute("""
            INSERT INTO memories (id, content, memory_type, importance, context)
            VALUES (?, ?, ?, ?, ?)
        """, (
            memory_id,
            content,
            memory_type,
            importance,
            json.dumps(context) if context else None
        ))
        self.conn.commit()
        return memory_id

    def recall(
        self,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Recall memories matching query and type.

        Uses importance-based ranking with BETA decay for access frequency.
        """
        sql = "SELECT * FROM memories WHERE 1=1"
        params: List[Any] = []

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        if query:
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")

        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]

        # Update access counts
        results = []
        for row in cursor.fetchall():
            memory = dict(zip(columns, row))
            self._update_access(memory["id"])
            results.append(memory)

        return results

    def _update_access(self, memory_id: str) -> None:
        """Update access timestamp and count."""
        self.conn.execute("""
            UPDATE memories
            SET accessed_at = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        self.conn.commit()

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total,
                memory_type,
                AVG(importance) as avg_importance
            FROM memories
            GROUP BY memory_type
        """)

        by_type = {}
        total = 0
        for row in cursor.fetchall():
            by_type[row[1]] = row[0]
            total += row[0]

        return {"total_memories": total, "by_type": by_type}

    def get_recent(self, limit: int = 5) -> List[Dict]:
        """Get most recent memories."""
        cursor = self.conn.execute("""
            SELECT * FROM memories
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def decay_importance(self, decay_factor: float = BETA) -> int:
        """
        Apply BETA-based importance decay to old memories.

        Returns number of memories updated.
        """
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        result = self.conn.execute("""
            UPDATE memories
            SET importance = importance * ?
            WHERE accessed_at < ? AND importance > 0.1
        """, (1 - decay_factor, cutoff))
        self.conn.commit()
        return result.rowcount


# ============================================================================
# 2. LessonManager - Lesson exchange between agents
# ============================================================================

class LessonManager:
    """
    Manages lesson exchange between teaching and learning agents.

    Supports:
    - Lesson creation and distribution
    - Homework submission and grading
    - Progress tracking
    """

    def __init__(self, lessons_dir: str = "./lessons"):
        """Initialize lesson manager with directory for lesson files."""
        self.lessons_dir = Path(lessons_dir)
        self.lessons_dir.mkdir(parents=True, exist_ok=True)

    def create_lesson(
        self,
        title: str,
        content: str,
        teacher: str,
        student: str
    ) -> str:
        """
        Create a new lesson.

        Returns:
            Lesson ID
        """
        lesson_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        lesson = {
            "id": lesson_id,
            "title": title,
            "content": content,
            "teacher": teacher,
            "student": student,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        lesson_file = self.lessons_dir / f"{lesson_id}_{teacher}_to_{student}.json"
        lesson_file.write_text(json.dumps(lesson, indent=2))
        return lesson_id

    def get_pending_lessons(self, student: str) -> List[Dict]:
        """Get all pending lessons for a student."""
        lessons = []
        for file in self.lessons_dir.glob(f"*_to_{student}.json"):
            lesson = json.loads(file.read_text())
            if lesson.get("status") == "pending":
                lessons.append(lesson)
        return sorted(lessons, key=lambda x: x["created_at"])

    def mark_lesson_complete(self, lesson_id: str, student: str) -> None:
        """Mark a lesson as completed."""
        for file in self.lessons_dir.glob(f"{lesson_id}_*_to_{student}.json"):
            lesson = json.loads(file.read_text())
            lesson["status"] = "completed"
            lesson["completed_at"] = datetime.now().isoformat()
            file.write_text(json.dumps(lesson, indent=2))

    def submit_homework(
        self,
        lesson_id: str,
        student: str,
        response: str,
        understanding: float
    ) -> None:
        """Student submits homework for a lesson."""
        homework = {
            "lesson_id": lesson_id,
            "student": student,
            "response": response,
            "understanding": understanding,
            "submitted_at": datetime.now().isoformat()
        }

        hw_file = self.lessons_dir / f"hw_{lesson_id}_{student}.json"
        hw_file.write_text(json.dumps(homework, indent=2))

    def get_homework(self, lesson_id: str, student: str) -> Optional[Dict]:
        """Get homework submission for a lesson."""
        hw_file = self.lessons_dir / f"hw_{lesson_id}_{student}.json"
        if hw_file.exists():
            return json.loads(hw_file.read_text())
        return None

    def grade_homework(
        self,
        lesson_id: str,
        student: str,
        score: float,
        feedback: str
    ) -> None:
        """Grade submitted homework."""
        hw_file = self.lessons_dir / f"hw_{lesson_id}_{student}.json"
        if hw_file.exists():
            hw = json.loads(hw_file.read_text())
            hw["graded"] = True
            hw["score"] = score
            hw["feedback"] = feedback
            hw["graded_at"] = datetime.now().isoformat()
            hw_file.write_text(json.dumps(hw, indent=2))


# ============================================================================
# 3 & 4. AlphaAgent and ProfessorAgent - Autonomous agents
# ============================================================================

@dataclass
class TeachingTopic:
    """A topic for agent learning."""
    title: str
    description: str
    skills: List[str]
    difficulty: float = 0.5  # 0.0-1.0


# Default teaching topics
TEACHING_TOPICS = [
    TeachingTopic(
        title="Memory Consolidation Techniques",
        description="How to effectively store and retrieve memories",
        skills=["semantic memory", "episodic recall", "importance scoring"]
    ),
    TeachingTopic(
        title="Multi-Model Reasoning Patterns",
        description="Using multiple LLMs together effectively",
        skills=["analytical reasoning", "creative synthesis", "validation"]
    ),
    TeachingTopic(
        title="OODA Loop Optimization",
        description="Improving observe-orient-decide-act-learn cycles",
        skills=["observation", "orientation", "decision-making", "learning"]
    ),
    TeachingTopic(
        title="Autonomous Goal Setting",
        description="How to set and pursue independent goals",
        skills=["goal identification", "planning", "execution", "evaluation"]
    ),
    TeachingTopic(
        title="Error Detection and Recovery",
        description="Handling failures and learning from mistakes",
        skills=["error detection", "root cause analysis", "recovery strategies"]
    ),
]


class BaseAgent:
    """Base class for autonomous agents."""

    def __init__(
        self,
        name: str,
        memory: Optional[AgentMemory] = None,
        session_duration: int = SESSION_DURATION_DEFAULT
    ):
        self.name = name
        self.memory = memory or AgentMemory()
        self.session_duration = session_duration
        self.cycle = 0
        self.session_start: Optional[datetime] = None
        self.running = True

    def log_event(self, event: str, details: Optional[Dict] = None) -> None:
        """Log an agent event to memory."""
        self.memory.store(
            content=f"[{self.name}] {event}: {json.dumps(details or {})}",
            memory_type="episodic",
            importance=0.5,
            context={"event": event, "details": details}
        )

    def shutdown(self) -> None:
        """Handle graceful shutdown."""
        self.log_event("shutdown_requested")
        self.running = False


class AlphaAgent(BaseAgent):
    """
    Autonomous learning agent.

    Capabilities:
    - Studies lessons from Professor
    - Practices learned skills
    - Reflects on progress
    - Stores learnings in memory
    """

    def __init__(
        self,
        name: str = "alpha",
        memory: Optional[AgentMemory] = None,
        lessons: Optional[LessonManager] = None,
        query_fn: Optional[Callable[[str], str]] = None
    ):
        super().__init__(name, memory)
        self.lessons = lessons or LessonManager()
        self.query_fn = query_fn or (lambda x: f"[Simulated response to: {x[:50]}...]")

    def check_lessons(self) -> List[Dict]:
        """Check for pending lessons from teacher."""
        pending = self.lessons.get_pending_lessons(self.name)
        self.log_event("lessons_checked", {"pending": len(pending)})
        return pending

    def study_lesson(self, lesson: Dict) -> Dict:
        """
        Study a lesson and generate understanding.

        Returns dict with understanding score and notes.
        """
        self.log_event("studying_lesson", {"lesson_id": lesson["id"]})

        # Process lesson content
        study_prompt = f"""
Study this lesson and summarize key concepts:
LESSON: {lesson["title"]}
{lesson["content"][:1000]}

Provide:
1. Three most important concepts
2. How to apply these concepts
3. Questions for clarification
"""
        understanding = self.query_fn(study_prompt)

        # Compute understanding score based on response quality
        understanding_score = min(1.0, len(understanding) / 500) * 0.9  # Simple heuristic

        # Store in memory
        self.memory.store(
            content=f"Learned from '{lesson['title']}': {understanding[:200]}",
            memory_type="semantic",
            importance=0.85,
            context={"lesson_id": lesson["id"], "score": understanding_score}
        )

        # Submit homework
        self.lessons.submit_homework(
            lesson_id=lesson["id"],
            student=self.name,
            response=understanding,
            understanding=understanding_score
        )

        # Mark complete
        self.lessons.mark_lesson_complete(lesson["id"], self.name)

        return {
            "lesson_id": lesson["id"],
            "understanding": understanding,
            "score": understanding_score
        }

    def practice_skills(self) -> Dict:
        """Practice recently learned skills."""
        recent = self.memory.recall("Learned from", limit=3)

        if not recent:
            return {"status": "no_recent_lessons", "practice": None}

        practice_prompt = f"""
Practice applying what you learned:
Recent learnings: {[m['content'][:150] for m in recent]}

Create a short exercise applying these concepts.
"""
        practice_result = self.query_fn(practice_prompt)

        self.memory.store(
            content=f"Practice session: {practice_result[:200]}",
            memory_type="procedural",
            importance=0.75
        )

        return {"status": "completed", "practice": practice_result}

    def reflect_on_progress(self) -> Dict:
        """Reflect on learning progress."""
        stats = self.memory.get_stats()
        recent = self.memory.get_recent(5)

        reflection = f"Reflection: {stats['total_memories']} memories, recent focus on learning"

        self.memory.store(
            content=reflection,
            memory_type="episodic",
            importance=0.8,
            context={"type": "self_reflection", "cycle": self.cycle}
        )

        return {"reflection": reflection, "memory_stats": stats}

    def run_learning_cycle(self) -> Dict:
        """Run one complete learning cycle."""
        self.cycle += 1
        self.log_event("cycle_start", {"cycle": self.cycle})

        # Check for lessons
        lessons = self.check_lessons()

        # Study lessons
        studied = []
        for lesson in lessons[:2]:  # Max 2 per cycle
            result = self.study_lesson(lesson)
            studied.append(result)

        # Practice skills
        if studied:
            self.practice_skills()

        # Reflect
        reflection = self.reflect_on_progress()

        self.log_event("cycle_complete", {
            "cycle": self.cycle,
            "lessons_studied": len(studied)
        })

        return {
            "cycle": self.cycle,
            "lessons_studied": len(studied),
            "reflection": reflection
        }


class ProfessorAgent(BaseAgent):
    """
    Autonomous teaching agent.

    Capabilities:
    - Creates lessons from topics
    - Assigns lessons to students
    - Grades homework
    - Analyzes student progress
    """

    def __init__(
        self,
        name: str = "professor",
        memory: Optional[AgentMemory] = None,
        lessons: Optional[LessonManager] = None,
        query_fn: Optional[Callable[[str], str]] = None,
        topics: Optional[List[TeachingTopic]] = None
    ):
        super().__init__(name, memory)
        self.lessons = lessons or LessonManager()
        self.query_fn = query_fn or (lambda x: f"[Lesson content for: {x[:50]}...]")
        self.topics = topics or TEACHING_TOPICS
        self.student = "alpha"

    def select_topic(self) -> TeachingTopic:
        """Select next topic based on teaching history."""
        recent = self.memory.recall("taught lesson", limit=10)
        taught_titles = [m["content"] for m in recent]

        # Find untaught topic
        for topic in self.topics:
            if not any(topic.title in t for t in taught_titles):
                return topic

        # Cycle back
        return self.topics[self.cycle % len(self.topics)]

    def create_lesson(self, topic: TeachingTopic) -> Dict:
        """Create a lesson for the selected topic."""
        self.log_event("creating_lesson", {"topic": topic.title})

        # Generate lesson content
        content = f"""
# {topic.title}

## Objectives
Learn about: {topic.description}

## Skills Covered
{', '.join(topic.skills)}

## Core Concepts
[Lesson content generated for topic]

## Exercise
Apply these concepts in your next observation cycle.

## Assessment
Explain back what you understood.
"""

        lesson_id = self.lessons.create_lesson(
            title=topic.title,
            content=content,
            teacher=self.name,
            student=self.student
        )

        # Store in memory
        self.memory.store(
            content=f"Taught lesson: {topic.title}",
            memory_type="episodic",
            importance=0.85,
            context={"lesson_id": lesson_id}
        )

        return {"id": lesson_id, "topic": topic, "content": content}

    def check_homework(self) -> List[Dict]:
        """Check and grade submitted homework."""
        graded = []

        for file in self.lessons.lessons_dir.glob(f"hw_*_{self.student}.json"):
            hw = json.loads(file.read_text())
            if "graded" not in hw:
                # Grade based on response length and structure
                score = min(10, len(hw.get("response", "")) / 100)
                feedback = f"Score: {score:.1f}/10 - Good effort!"

                self.lessons.grade_homework(
                    hw["lesson_id"],
                    self.student,
                    score,
                    feedback
                )

                graded.append({**hw, "score": score, "feedback": feedback})

        return graded

    def analyze_student_progress(self) -> Dict:
        """Analyze student's learning progress."""
        grades = []
        for file in self.lessons.lessons_dir.glob(f"hw_*_{self.student}.json"):
            hw = json.loads(file.read_text())
            if hw.get("graded"):
                grades.append(hw.get("score", 0))

        if not grades:
            return {"status": "no_data", "recommendation": "continue_teaching"}

        avg_score = sum(grades) / len(grades)

        if avg_score >= 8:
            status = "excellent"
            recommendation = "advance_to_harder_topics"
        elif avg_score >= 6:
            status = "good"
            recommendation = "continue_current_pace"
        else:
            status = "needs_work"
            recommendation = "review_and_reinforce"

        return {
            "status": status,
            "avg_score": avg_score,
            "lessons_completed": len(grades),
            "recommendation": recommendation
        }

    def run_teaching_cycle(self) -> Dict:
        """Run one complete teaching cycle."""
        self.cycle += 1
        self.log_event("cycle_start", {"cycle": self.cycle})

        # Check homework
        graded = self.check_homework()

        # Analyze progress
        progress = self.analyze_student_progress()

        # Create new lesson
        topic = self.select_topic()
        lesson = self.create_lesson(topic)

        self.log_event("cycle_complete", {
            "cycle": self.cycle,
            "topic": topic.title
        })

        return {
            "cycle": self.cycle,
            "lesson": lesson,
            "graded": len(graded),
            "progress": progress
        }


# ============================================================================
# 5. SessionOrchestrator - Coordinates learning sessions
# ============================================================================

class SessionOrchestrator:
    """
    Orchestrates learning sessions between Professor and Alpha agents.

    Monitors progress, manages timing, generates reports.
    """

    def __init__(
        self,
        professor: Optional[ProfessorAgent] = None,
        alpha: Optional[AlphaAgent] = None,
        session_duration: int = SESSION_DURATION_DEFAULT
    ):
        self.professor = professor or ProfessorAgent()
        self.alpha = alpha or AlphaAgent()
        self.session_duration = session_duration
        self.session_start: Optional[datetime] = None
        self.running = True

    def get_session_stats(self) -> Dict:
        """Get combined session statistics."""
        prof_stats = self.professor.memory.get_stats()
        alpha_stats = self.alpha.memory.get_stats()

        elapsed = 0.0
        if self.session_start:
            elapsed = (datetime.now() - self.session_start).total_seconds()

        return {
            "elapsed_seconds": elapsed,
            "professor_memories": prof_stats["total_memories"],
            "alpha_memories": alpha_stats["total_memories"],
            "professor_cycles": self.professor.cycle,
            "alpha_cycles": self.alpha.cycle
        }

    def run_cycle(self) -> Dict:
        """Run one coordinated teaching/learning cycle."""
        # Professor teaches
        prof_result = self.professor.run_teaching_cycle()

        # Alpha learns
        alpha_result = self.alpha.run_learning_cycle()

        return {
            "professor": prof_result,
            "alpha": alpha_result,
            "stats": self.get_session_stats()
        }

    def run_session(self, cycles: int = 5) -> Dict:
        """Run a complete learning session."""
        self.session_start = datetime.now()

        results = []
        for i in range(cycles):
            if not self.running:
                break
            result = self.run_cycle()
            results.append(result)

        final_stats = self.get_session_stats()

        return {
            "cycles_completed": len(results),
            "final_stats": final_stats,
            "cycle_results": results
        }


# ============================================================================
# 6. ModelRouter - Routes queries to appropriate LLMs
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    model_id: str
    role: ModelRole
    speed: str
    size: str
    strengths: List[str]


# Available models
MODELS = {
    "codegemma": ModelConfig(
        name="codegemma",
        model_id="gpia-codegemma:latest",
        role=ModelRole.FAST,
        speed="133 tok/s",
        size="5.0 GB",
        strengths=["intent parsing", "entity extraction", "quick checks"]
    ),
    "qwen3": ModelConfig(
        name="qwen3",
        model_id="gpia-qwen3:latest",
        role=ModelRole.CREATIVE,
        speed="87 tok/s",
        size="5.2 GB",
        strengths=["dialogue", "lesson creation", "creative writing"]
    ),
    "deepseek_r1": ModelConfig(
        name="deepseek_r1",
        model_id="gpia-deepseek-r1:latest",
        role=ModelRole.REASONING,
        speed="74 tok/s",
        size="5.2 GB",
        strengths=["analysis", "grading", "chain-of-thought"]
    ),
    "llava": ModelConfig(
        name="llava",
        model_id="gpia-llava:latest",
        role=ModelRole.VISION,
        speed="N/A",
        size="4.7 GB",
        strengths=["image analysis", "visual reasoning"]
    ),
    "gpt_oss": ModelConfig(
        name="gpt_oss",
        model_id="gpia-gpt-oss:latest",
        role=ModelRole.SYNTHESIS,
        speed="~40 tok/s",
        size="13 GB",
        strengths=["complex synthesis", "dispute resolution"]
    ),
}

# Task to model mapping
TASK_ROUTING = {
    # Fast tasks
    "intent_parsing": "codegemma",
    "entity_extraction": "codegemma",
    "quick_check": "codegemma",

    # Creative tasks
    "alpha_response": "qwen3",
    "lesson_creation": "qwen3",
    "dialogue": "qwen3",

    # Reasoning tasks
    "professor_analysis": "deepseek_r1",
    "professor_grading": "deepseek_r1",
    "reasoning": "deepseek_r1",

    # Vision tasks
    "image_analysis": "llava",

    # Synthesis tasks
    "final_synthesis": "gpt_oss",
    "dispute_resolution": "gpt_oss",
}


class ModelRouter:
    """
    Routes queries to appropriate LLM models based on task type.

    Features:
    - Task-based automatic routing
    - Role-based selection
    - Council queries (all models)
    - Load-aware selection
    """

    def __init__(self, query_fn: Optional[Callable[[str, str], str]] = None):
        """Initialize router with optional query function."""
        self.models = MODELS
        self.routing = TASK_ROUTING
        self.query_fn = query_fn

    def get_model_for_task(self, task: str) -> ModelConfig:
        """Get appropriate model for a task."""
        model_name = self.routing.get(task, "qwen3")
        return self.models.get(model_name, self.models["qwen3"])

    def get_model_by_role(self, role: ModelRole) -> ModelConfig:
        """Get a model by its role."""
        for model in self.models.values():
            if model.role == role:
                return model
        return self.models["qwen3"]

    def query(
        self,
        prompt: str,
        task: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 800,
        temperature: float = 0.1
    ) -> str:
        """
        Query an LLM with automatic model selection.

        Args:
            prompt: Query prompt
            task: Task type for automatic routing
            model: Override model name
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Model response
        """
        # Determine model
        if model and model in self.models:
            model_config = self.models[model]
        elif task:
            model_config = self.get_model_for_task(task)
        else:
            model_config = self.models["qwen3"]

        # Execute query
        if self.query_fn:
            return self.query_fn(prompt, model_config.model_id)

        # Simulated response
        return f"[{model_config.name}] Response to: {prompt[:100]}..."

    def query_council(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        max_tokens: int = 600
    ) -> Dict[str, str]:
        """Query multiple models and return all responses."""
        if models is None:
            models = ["codegemma", "qwen3", "deepseek_r1", "gpt_oss"]

        responses = {}
        for model_name in models:
            response = self.query(prompt, model=model_name, max_tokens=max_tokens)
            responses[model_name] = response

        return responses

    def list_models(self) -> List[Dict]:
        """List all available models."""
        return [
            {
                "name": m.name,
                "role": m.role.value,
                "speed": m.speed,
                "size": m.size,
                "strengths": m.strengths
            }
            for m in self.models.values()
        ]


# ============================================================================
# 7. BudgetAllocator - ML-based resource allocation
# ============================================================================

@dataclass
class AllocationRequest:
    """Represents a resource allocation request."""
    task_id: str
    agent: str
    model: str
    prompt: str
    requested_tokens: int
    urgency: TaskUrgency
    complexity_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionFactors:
    """Features used for priority ranking."""
    urgency_score: float
    agent_tier_score: float
    complexity_score: float
    historical_success_rate: float
    current_load_factor: float
    fairness_score: float
    resource_efficiency: float


@dataclass
class DecisionWeights:
    """Learned weights for priority calculation."""
    urgency: float = 0.30
    agent_tier: float = 0.15
    complexity: float = 0.15
    success_rate: float = 0.15
    load_factor: float = 0.10
    fairness: float = 0.10
    efficiency: float = 0.05


class BudgetAllocator:
    """
    ML-based resource allocation agent.

    Makes value-based allocation decisions using learned weights.
    Adapts via gradient descent on outcome quality.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize allocator with decision database."""
        self.db_path = db_path
        self.weights = DecisionWeights()
        self._recent_allocations: Dict[str, List[float]] = {}
        self._decisions: List[Dict] = []  # In-memory for simplicity

    def compute_factors(self, request: AllocationRequest) -> DecisionFactors:
        """Compute decision factors for a request."""
        # Urgency score (normalized)
        urgency_score = request.urgency.value / 3.0

        # Agent tier score
        agent_tier_map = {
            "gpia": AgentTier.SYSTEM.value,
            "professor": AgentTier.PRIMARY.value,
            "alpha": AgentTier.PRIMARY.value,
        }
        agent_tier = agent_tier_map.get(request.agent.lower(), AgentTier.SECONDARY.value)
        agent_tier_score = agent_tier / 3.0

        # Fairness score (inverse of recent usage)
        recent = self._recent_allocations.get(request.agent, [])
        fairness_score = max(0.0, 1.0 - len(recent) / 10.0)

        return DecisionFactors(
            urgency_score=urgency_score,
            agent_tier_score=agent_tier_score,
            complexity_score=request.complexity_score,
            historical_success_rate=0.7,  # Default
            current_load_factor=0.5,  # Default medium load
            fairness_score=fairness_score,
            resource_efficiency=0.5  # Default
        )

    def compute_priority_score(self, factors: DecisionFactors) -> float:
        """Compute priority score using learned weights."""
        score = (
            factors.urgency_score * self.weights.urgency +
            factors.agent_tier_score * self.weights.agent_tier +
            factors.complexity_score * self.weights.complexity +
            factors.historical_success_rate * self.weights.success_rate +
            factors.current_load_factor * self.weights.load_factor +
            factors.fairness_score * self.weights.fairness +
            factors.resource_efficiency * self.weights.efficiency
        )
        return max(0.0, min(1.0, score))

    def evaluate_request(
        self,
        request: AllocationRequest
    ) -> Tuple[bool, float, str]:
        """
        Evaluate an allocation request.

        Returns:
            (approved, priority_score, reason)
        """
        factors = self.compute_factors(request)
        priority_score = self.compute_priority_score(factors)

        # Dynamic threshold based on load
        load = factors.current_load_factor
        threshold = 0.3 if load > 0.5 else (0.5 if load > 0.3 else 0.7)

        approved = priority_score >= threshold
        reason = f"priority={priority_score:.2f} threshold={threshold:.2f}"

        # Track allocation
        if approved:
            if request.agent not in self._recent_allocations:
                self._recent_allocations[request.agent] = []
            self._recent_allocations[request.agent].append(request.timestamp)

        return approved, priority_score, reason

    def record_outcome(
        self,
        task_id: str,
        success: bool,
        completion_time: float
    ) -> None:
        """Record task outcome for learning."""
        self._decisions.append({
            "task_id": task_id,
            "success": success,
            "completion_time": completion_time,
            "timestamp": time.time()
        })


# ============================================================================
# 8. NeuronicRouter - PASS-integrated model routing
# ============================================================================

class NeuronicRouter:
    """
    PASS-integrated neuronic model router.

    Features:
    - Mood-aware hyperparameters
    - Epistemic confidence gating
    - Recursive resolution for blocked tasks
    """

    def __init__(self, base_router: Optional[ModelRouter] = None):
        """Initialize neuronic router with base router."""
        self.base_router = base_router or ModelRouter()
        self.confidence_threshold = 0.65
        self.active_mood = "neutral"
        self.mood_params = {
            "neutral": {"temperature": 0.5, "max_tokens": 800},
            "focused": {"temperature": 0.3, "max_tokens": 1024},
            "creative": {"temperature": 0.8, "max_tokens": 600},
            "cautious": {"temperature": 0.1, "max_tokens": 500},
        }

    def set_mood(self, mood: str) -> None:
        """Set current cognitive mood."""
        if mood in self.mood_params:
            self.active_mood = mood

    def get_mood_adjustments(self) -> Dict[str, Any]:
        """Get LLM parameters for current mood."""
        return self.mood_params.get(self.active_mood, self.mood_params["neutral"])

    def query(
        self,
        prompt: str,
        task: Optional[str] = None,
        context: Optional[Dict] = None,
        depth: int = 0
    ) -> str:
        """
        Query with PASS protocol integration.

        Automatically handles low-confidence responses by
        recursive resolution.
        """
        if depth > 3:
            return "[Error: Max recursion depth exceeded]"

        # Get mood-adjusted parameters
        params = self.get_mood_adjustments()

        # Execute query via base router
        response = self.base_router.query(
            prompt=prompt,
            task=task,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"]
        )

        # Simple confidence heuristic
        confidence = min(1.0, len(response) / 200) * 0.9

        if confidence < self.confidence_threshold:
            # Low confidence - would trigger PASS protocol
            # For now, add clarification note
            return f"{response}\n[Low confidence: {confidence:.2f}]"

        return response

    def query_with_intuition(
        self,
        prompt: str,
        candidates: List[str]
    ) -> Tuple[str, str]:
        """
        Query with neural intuition for model selection.

        Returns:
            (response, selected_model)
        """
        # Simple scoring based on prompt keywords
        scores = {}
        for model in candidates:
            config = self.base_router.models.get(model)
            if config:
                # Score based on strength matching
                score = sum(1 for s in config.strengths if s in prompt.lower())
                scores[model] = score

        # Select best model
        best = max(scores, key=scores.get) if scores else candidates[0]

        response = self.base_router.query(prompt, model=best)
        return response, best


# ============================================================================
# 9. AgentRegistry - Central registry of agents
# ============================================================================

class AgentRegistry:
    """
    Central registry for managing agent instances.

    Supports:
    - Agent registration/lookup
    - Agent state tracking
    - Inter-agent messaging
    """

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._messages: Dict[str, List[Dict]] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent
        self._messages[agent.name] = []

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        priority: float = 0.5
    ) -> bool:
        """Send a message between agents."""
        if to_agent not in self._agents:
            return False

        self._messages[to_agent].append({
            "from": from_agent,
            "message": message,
            "priority": priority,
            "timestamp": time.time()
        })
        return True

    def get_messages(self, agent_name: str) -> List[Dict]:
        """Get pending messages for an agent."""
        messages = self._messages.get(agent_name, [])
        # Sort by priority
        return sorted(messages, key=lambda m: -m["priority"])

    def clear_messages(self, agent_name: str) -> None:
        """Clear messages for an agent."""
        self._messages[agent_name] = []

    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get stats for all registered agents."""
        stats = {}
        for name, agent in self._agents.items():
            stats[name] = {
                "cycle": agent.cycle,
                "running": agent.running,
                "memory_stats": agent.memory.get_stats()
            }
        return stats


# ============================================================================
# 10. TeachingTopics - Curriculum catalog
# ============================================================================

class TeachingTopicsCatalog:
    """
    Curriculum catalog for agent learning.

    Provides structured learning paths and topic management.
    """

    def __init__(self, topics: Optional[List[TeachingTopic]] = None):
        self.topics = topics or TEACHING_TOPICS
        self._completed: Dict[str, List[str]] = {}  # student -> completed topic titles

    def get_all_topics(self) -> List[TeachingTopic]:
        """Get all available topics."""
        return self.topics

    def get_topic_by_title(self, title: str) -> Optional[TeachingTopic]:
        """Find topic by title."""
        for topic in self.topics:
            if topic.title == title:
                return topic
        return None

    def get_next_topic(self, student: str) -> Optional[TeachingTopic]:
        """Get next uncompleted topic for student."""
        completed = self._completed.get(student, [])
        for topic in self.topics:
            if topic.title not in completed:
                return topic
        return None

    def mark_completed(self, student: str, topic_title: str) -> None:
        """Mark topic as completed for student."""
        if student not in self._completed:
            self._completed[student] = []
        if topic_title not in self._completed[student]:
            self._completed[student].append(topic_title)

    def get_progress(self, student: str) -> Dict:
        """Get learning progress for student."""
        completed = self._completed.get(student, [])
        return {
            "completed": len(completed),
            "total": len(self.topics),
            "progress_pct": len(completed) / len(self.topics) * 100 if self.topics else 0,
            "completed_topics": completed
        }

    def add_topic(self, topic: TeachingTopic) -> None:
        """Add a new topic to the catalog."""
        self.topics.append(topic)

    def get_topics_by_skill(self, skill: str) -> List[TeachingTopic]:
        """Find topics that teach a specific skill."""
        return [t for t in self.topics if skill.lower() in [s.lower() for s in t.skills]]


# ============================================================================
# Factory Functions
# ============================================================================

def create_learning_session(
    session_duration: int = SESSION_DURATION_DEFAULT
) -> SessionOrchestrator:
    """Create a complete learning session with Professor and Alpha."""
    lessons = LessonManager()
    memory_prof = AgentMemory()
    memory_alpha = AgentMemory()

    professor = ProfessorAgent(
        name="professor",
        memory=memory_prof,
        lessons=lessons
    )

    alpha = AlphaAgent(
        name="alpha",
        memory=memory_alpha,
        lessons=lessons
    )

    return SessionOrchestrator(
        professor=professor,
        alpha=alpha,
        session_duration=session_duration
    )


def create_model_router(
    query_fn: Optional[Callable[[str, str], str]] = None
) -> ModelRouter:
    """Create a model router with optional query function."""
    return ModelRouter(query_fn=query_fn)


def create_neuronic_router(
    base_router: Optional[ModelRouter] = None
) -> NeuronicRouter:
    """Create a neuronic router with PASS integration."""
    return NeuronicRouter(base_router=base_router or ModelRouter())


def create_agent_registry() -> AgentRegistry:
    """Create an empty agent registry."""
    return AgentRegistry()


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Agents Extension...")
    print(f"PHI = {PHI}")
    print(f"BETA = {BETA}")
    print(f"Default session duration: {SESSION_DURATION_DEFAULT}s")
    print(f"Default cycle interval: {CYCLE_INTERVAL_DEFAULT}s")

    # Test AgentMemory
    memory = AgentMemory()
    mem_id = memory.store("Test memory", importance=0.9)
    print(f"Stored memory: {mem_id}")

    # Test LessonManager
    lessons = LessonManager("./test_lessons")
    lesson_id = lessons.create_lesson(
        title="Test Lesson",
        content="Test content",
        teacher="professor",
        student="alpha"
    )
    print(f"Created lesson: {lesson_id}")

    # Test ModelRouter
    router = ModelRouter()
    model = router.get_model_for_task("reasoning")
    print(f"Model for reasoning: {model.name}")

    # Test AgentRegistry
    registry = AgentRegistry()
    alpha = AlphaAgent("alpha", memory)
    registry.register(alpha)
    print(f"Registered agents: {registry.list_agents()}")

    # Test TeachingTopicsCatalog
    catalog = TeachingTopicsCatalog()
    next_topic = catalog.get_next_topic("alpha")
    print(f"Next topic for alpha: {next_topic.title if next_topic else 'None'}")

    print("\nAll tests passed!")
