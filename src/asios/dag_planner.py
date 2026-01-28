"""
DAG Planner - Task Decomposition

Converts vague requests into strict DAG of atomic steps.
Provides structured execution plans with approval gates.

Features:
- Task decomposition into atomic steps
- DAG generation with dependencies
- Approval gate support
- Plan deduplication via hashing
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path


class TaskStatus(Enum):
    """Status of a task in the plan."""
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """An atomic task in the execution plan."""
    id: str
    name: str
    description: str
    action: str  # The action to perform
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

    # Metadata
    estimated_cost: str = "low"  # low, medium, high
    requires_approval: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "estimated_cost": self.estimated_cost,
            "requires_approval": self.requires_approval,
        }


@dataclass
class ExecutionPlan:
    """A DAG of tasks forming an execution plan."""
    id: str
    name: str
    description: str
    tasks: List[Task] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    approved: bool = False
    approved_at: Optional[float] = None

    def add_task(self, task: Task):
        """Add a task to the plan."""
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all deps completed)."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}

        ready = []
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)

        return ready

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.tasks
        )

    def get_hash(self) -> str:
        """Get deterministic hash of the plan."""
        content = json.dumps(
            [t.to_dict() for t in self.tasks],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at,
            "approved": self.approved,
            "hash": self.get_hash(),
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid diagram of the DAG."""
        lines = ["graph TD"]
        for task in self.tasks:
            # Node
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.APPROVED: "✅",
                TaskStatus.RUNNING: "🔄",
                TaskStatus.COMPLETED: "✓",
                TaskStatus.FAILED: "❌",
                TaskStatus.SKIPPED: "⏭",
            }.get(task.status, "")
            lines.append(f'    {task.id}["{status_icon} {task.name}"]')

            # Edges
            for dep in task.dependencies:
                lines.append(f"    {dep} --> {task.id}")

        return "\n".join(lines)


class DAGPlanner:
    """
    DAG Planner for Task Decomposition.

    Converts high-level requests into executable task graphs
    with proper dependency ordering.

    Example:
        planner = DAGPlanner()

        plan = planner.plan("Organize my downloads folder")
        print(plan.to_mermaid())

        # Approve and execute
        planner.approve(plan.id)
        while not plan.is_complete():
            ready = plan.get_ready_tasks()
            for task in ready:
                result = await execute(task)
                planner.complete_task(plan.id, task.id, result)
    """

    # Common task patterns
    PATTERNS = {
        "organize": [
            ("scan", "Scan directory contents", "list_files"),
            ("analyze", "Analyze file types", "categorize_files"),
            ("plan_moves", "Plan file movements", "generate_moves"),
            ("execute_moves", "Execute file movements", "move_files"),
        ],
        "backup": [
            ("identify", "Identify files to backup", "list_files"),
            ("compress", "Compress files", "create_archive"),
            ("transfer", "Transfer to backup location", "copy_files"),
            ("verify", "Verify backup integrity", "checksum_verify"),
        ],
        "deploy": [
            ("build", "Build application", "run_build"),
            ("test", "Run tests", "run_tests"),
            ("package", "Create deployment package", "create_package"),
            ("deploy", "Deploy to target", "deploy_package"),
        ],
    }

    def __init__(
        self,
        max_depth: int = 1,
        approval_gate_path: Optional[Path] = None,
    ):
        self.max_depth = max_depth
        self.approval_gate_path = approval_gate_path
        self.plans: Dict[str, ExecutionPlan] = {}

        # Statistics
        self.total_plans = 0
        self.approved_plans = 0

    def plan(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for a request.

        Args:
            request: High-level request description
            context: Optional context information

        Returns:
            ExecutionPlan with task DAG
        """
        self.total_plans += 1

        # Generate plan ID
        plan_id = f"plan_{self.total_plans:04d}"

        plan = ExecutionPlan(
            id=plan_id,
            name=f"Plan for: {request[:50]}",
            description=request,
        )

        # Check for pattern match
        request_lower = request.lower()
        matched_pattern = None
        for pattern_name, tasks in self.PATTERNS.items():
            if pattern_name in request_lower:
                matched_pattern = tasks
                break

        if matched_pattern:
            # Use pattern
            prev_id = None
            for i, (name, desc, action) in enumerate(matched_pattern):
                task_id = f"{plan_id}_task_{i:02d}"
                task = Task(
                    id=task_id,
                    name=name,
                    description=desc,
                    action=action,
                    dependencies=[prev_id] if prev_id else [],
                )
                plan.add_task(task)
                prev_id = task_id
        else:
            # Generic decomposition
            tasks = self._decompose_generic(request, context)
            prev_id = None
            for i, (name, desc, action) in enumerate(tasks):
                task_id = f"{plan_id}_task_{i:02d}"
                task = Task(
                    id=task_id,
                    name=name,
                    description=desc,
                    action=action,
                    dependencies=[prev_id] if prev_id else [],
                )
                plan.add_task(task)
                prev_id = task_id

        self.plans[plan_id] = plan
        return plan

    def _decompose_generic(
        self,
        request: str,
        context: Optional[Dict[str, Any]],
    ) -> List[tuple]:
        """Generic task decomposition."""
        # Simple heuristic decomposition
        return [
            ("analyze", f"Analyze request: {request[:30]}...", "analyze_request"),
            ("prepare", "Prepare resources", "prepare_resources"),
            ("execute", "Execute main action", "execute_action"),
            ("verify", "Verify results", "verify_results"),
        ]

    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)

    def approve(self, plan_id: str) -> bool:
        """Approve a plan for execution."""
        plan = self.get_plan(plan_id)
        if not plan:
            return False

        plan.approved = True
        plan.approved_at = time.time()
        self.approved_plans += 1

        # Mark tasks as approved
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.APPROVED

        # Write to approval gate if configured
        if self.approval_gate_path:
            self._write_approval(plan)

        return True

    def _write_approval(self, plan: ExecutionPlan):
        """Write approval to gate file."""
        self.approval_gate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.approval_gate_path, "w") as f:
            json.dump({
                "plan_id": plan.id,
                "approved_at": plan.approved_at,
                "hash": plan.get_hash(),
            }, f)

    def start_task(self, plan_id: str, task_id: str) -> bool:
        """Mark a task as running."""
        plan = self.get_plan(plan_id)
        if not plan:
            return False

        task = plan.get_task(task_id)
        if not task or task.status != TaskStatus.APPROVED:
            return False

        task.status = TaskStatus.RUNNING
        return True

    def complete_task(
        self,
        plan_id: str,
        task_id: str,
        result: Any,
    ) -> bool:
        """Mark a task as completed."""
        plan = self.get_plan(plan_id)
        if not plan:
            return False

        task = plan.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.result = result
        return True

    def fail_task(
        self,
        plan_id: str,
        task_id: str,
        error: str,
    ) -> bool:
        """Mark a task as failed."""
        plan = self.get_plan(plan_id)
        if not plan:
            return False

        task = plan.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.FAILED
        task.error = error
        return True

    def stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "total_plans": self.total_plans,
            "approved_plans": self.approved_plans,
            "active_plans": len(self.plans),
            "approval_rate": self.approved_plans / max(1, self.total_plans),
        }
