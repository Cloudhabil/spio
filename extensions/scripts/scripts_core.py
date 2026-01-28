"""
Scripts Extension - Utility Script Registry and Runner

Catalog of 150+ utility scripts organized by category:

Categories (12):
1. Research (20)     - BSD, Riemann, research orchestration
2. Admin (15)        - System administration, policy
3. Agents (15)       - Agent orchestration, training
4. Validation (15)   - Integrity checks, verification
5. Data (15)         - Data processing, dense state
6. Network (10)      - Network diagnostics, monitoring
7. Build (10)        - CI/CD, compilation, optimization
8. Learning (15)     - ML training, fine-tuning
9. Genesis (10)      - Initialization, birth cycles
10. Manifold (10)    - Topology, manifold operations
11. Gardener (10)    - Filesystem, maintenance
12. Misc (15)        - Various utilities

Provides:
- ScriptCatalog: Registry of all scripts
- ScriptRunner: Execute scripts safely
- ScriptCategory: Categorization enum
- ScriptInfo: Script metadata

Reference: CLI-main/scripts/*
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ============================================================================
# ENUMS
# ============================================================================

class ScriptCategory(Enum):
    """Script categories."""
    RESEARCH = "research"
    ADMIN = "admin"
    AGENTS = "agents"
    VALIDATION = "validation"
    DATA = "data"
    NETWORK = "network"
    BUILD = "build"
    LEARNING = "learning"
    GENESIS = "genesis"
    MANIFOLD = "manifold"
    GARDENER = "gardener"
    MISC = "misc"


class ScriptStatus(Enum):
    """Script execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# SCRIPT INFO
# ============================================================================

@dataclass
class ScriptInfo:
    """Information about a script."""
    name: str
    category: ScriptCategory
    description: str
    path: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: ScriptStatus = ScriptStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "path": self.path,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "status": self.status.value,
        }


# ============================================================================
# SCRIPT CATALOG
# ============================================================================

# Research scripts (BSD, Riemann, research)
RESEARCH_SCRIPTS = [
    ScriptInfo("run_bsd_completion", ScriptCategory.RESEARCH, "Complete BSD conjecture validation", tags=["bsd", "math"]),
    ScriptInfo("run_bsd_gap_closure_framework", ScriptCategory.RESEARCH, "BSD gap closure framework", tags=["bsd"]),
    ScriptInfo("rh_2hour_discovery_sprint", ScriptCategory.RESEARCH, "2-hour RH discovery sprint", tags=["riemann"]),
    ScriptInfo("rh_extended_zero_verification", ScriptCategory.RESEARCH, "Extended RH zero verification", tags=["riemann"]),
    ScriptInfo("rh_verify_rmt_operator_theory", ScriptCategory.RESEARCH, "RH RMT operator theory", tags=["riemann"]),
    ScriptInfo("start_rh_research_ensemble", ScriptCategory.RESEARCH, "Start RH research ensemble", tags=["riemann"]),
    ScriptInfo("start_rh_adaptive_ensemble", ScriptCategory.RESEARCH, "Start RH adaptive ensemble", tags=["riemann"]),
    ScriptInfo("phase2_hamiltonian_construction", ScriptCategory.RESEARCH, "Phase 2 Hamiltonian construction", tags=["math"]),
    ScriptInfo("phase2b_hamiltonian_optimization", ScriptCategory.RESEARCH, "Phase 2b Hamiltonian optimization", tags=["math"]),
    ScriptInfo("phase2c_dlsfh_hamiltonian", ScriptCategory.RESEARCH, "Phase 2c DLSFH Hamiltonian", tags=["math"]),
    ScriptInfo("proof_model_selection", ScriptCategory.RESEARCH, "Proof model selection", tags=["math"]),
    ScriptInfo("find_optimal_cycle", ScriptCategory.RESEARCH, "Find optimal research cycle", tags=["optimization"]),
]

# Admin scripts
ADMIN_SCRIPTS = [
    ScriptInfo("admin_policy", ScriptCategory.ADMIN, "Admin policy management", tags=["policy"]),
    ScriptInfo("admin_tui", ScriptCategory.ADMIN, "Admin TUI interface", tags=["ui"]),
    ScriptInfo("runtime_compliance_shaper", ScriptCategory.ADMIN, "Runtime compliance shaping", tags=["compliance"]),
    ScriptInfo("self_patch", ScriptCategory.ADMIN, "Self-patching system", tags=["maintenance"]),
    ScriptInfo("placeholder_scanner", ScriptCategory.ADMIN, "Scan for placeholders", tags=["scan"]),
    ScriptInfo("kb_report", ScriptCategory.ADMIN, "Knowledge base report", tags=["report"]),
    ScriptInfo("macro_register", ScriptCategory.ADMIN, "Macro registration", tags=["macros"]),
    ScriptInfo("synapse_register", ScriptCategory.ADMIN, "Synapse registration", tags=["synapses"]),
]

# Agent scripts
AGENTS_SCRIPTS = [
    ScriptInfo("orchestrate_alpha_professor", ScriptCategory.AGENTS, "Orchestrate Alpha-Professor", tags=["agents"]),
    ScriptInfo("create_professor_now", ScriptCategory.AGENTS, "Create Professor agent", tags=["agents"]),
    ScriptInfo("start_meta_professor", ScriptCategory.AGENTS, "Start Meta-Professor", tags=["agents"]),
    ScriptInfo("professor_competition", ScriptCategory.AGENTS, "Professor competition", tags=["agents"]),
    ScriptInfo("loop_agent", ScriptCategory.AGENTS, "Loop agent execution", tags=["agents"]),
    ScriptInfo("provision_agent", ScriptCategory.AGENTS, "Provision new agent", tags=["agents"]),
    ScriptInfo("step1_registry_agent", ScriptCategory.AGENTS, "Registry agent step 1", tags=["agents"]),
    ScriptInfo("autonomous_hunting_daemon", ScriptCategory.AGENTS, "Autonomous hunting daemon", tags=["agents"]),
    ScriptInfo("auto_dev_agent", ScriptCategory.AGENTS, "Auto dev agent", tags=["agents"]),
]

# Validation scripts
VALIDATION_SCRIPTS = [
    ScriptInfo("verify_adaptive_ensemble_setup", ScriptCategory.VALIDATION, "Verify ensemble setup", tags=["verify"]),
    ScriptInfo("validate_active_skills", ScriptCategory.VALIDATION, "Validate active skills", tags=["skills"]),
    ScriptInfo("validate_manifold_integrity", ScriptCategory.VALIDATION, "Validate manifold integrity", tags=["manifold"]),
    ScriptInfo("run_epistemic_scan", ScriptCategory.VALIDATION, "Run epistemic scan", tags=["scan"]),
    ScriptInfo("audit_genesis_run", ScriptCategory.VALIDATION, "Audit genesis run", tags=["audit"]),
    ScriptInfo("micro_tune_auditor", ScriptCategory.VALIDATION, "Micro-tune auditor", tags=["audit"]),
    ScriptInfo("agi_certification", ScriptCategory.VALIDATION, "AGI certification", tags=["certification"]),
]

# Data scripts
DATA_SCRIPTS = [
    ScriptInfo("generate_dense_state", ScriptCategory.DATA, "Generate dense state", tags=["dense"]),
    ScriptInfo("dense_state_patch", ScriptCategory.DATA, "Patch dense state", tags=["dense"]),
    ScriptInfo("dense_state_proof", ScriptCategory.DATA, "Dense state proof", tags=["dense"]),
    ScriptInfo("lock_dense_state_stats", ScriptCategory.DATA, "Lock dense state stats", tags=["dense"]),
    ScriptInfo("train_dense_state_golden_route", ScriptCategory.DATA, "Train golden route", tags=["dense", "ml"]),
    ScriptInfo("store_session_context", ScriptCategory.DATA, "Store session context", tags=["context"]),
    ScriptInfo("recall_session", ScriptCategory.DATA, "Recall session", tags=["context"]),
    ScriptInfo("substrate_compression_report", ScriptCategory.DATA, "Substrate compression report", tags=["report"]),
]

# Network scripts
NETWORK_SCRIPTS = [
    ScriptInfo("run_network_diagnostics", ScriptCategory.NETWORK, "Run network diagnostics", tags=["network"]),
    ScriptInfo("brute_force_ping", ScriptCategory.NETWORK, "Brute force ping", tags=["network"]),
    ScriptInfo("scout_hidden_nodes", ScriptCategory.NETWORK, "Scout hidden nodes", tags=["network"]),
    ScriptInfo("hyper_sensory_scan", ScriptCategory.NETWORK, "Hyper-sensory scan", tags=["scan"]),
]

# Build scripts
BUILD_SCRIPTS = [
    ScriptInfo("cicd_runner", ScriptCategory.BUILD, "CI/CD runner", tags=["ci"]),
    ScriptInfo("build_livelink_ui", ScriptCategory.BUILD, "Build LiveLink UI", tags=["build"]),
    ScriptInfo("build_skill_index", ScriptCategory.BUILD, "Build skill index", tags=["skills"]),
    ScriptInfo("update_shell_skill_index", ScriptCategory.BUILD, "Update shell skill index", tags=["skills"]),
    ScriptInfo("fix_py311_escapes", ScriptCategory.BUILD, "Fix Python 3.11 escapes", tags=["fix"]),
    ScriptInfo("migrate_models_math_optimized", ScriptCategory.BUILD, "Migrate math-optimized models", tags=["migration"]),
]

# Learning scripts
LEARNING_SCRIPTS = [
    ScriptInfo("finetune_rh_models", ScriptCategory.LEARNING, "Fine-tune RH models", tags=["ml", "riemann"]),
    ScriptInfo("run_lora_learning_mission", ScriptCategory.LEARNING, "Run LoRA learning mission", tags=["ml"]),
    ScriptInfo("learn_continuous_operation", ScriptCategory.LEARNING, "Learn continuous operation", tags=["ml"]),
    ScriptInfo("teach_dialogue_skill", ScriptCategory.LEARNING, "Teach dialogue skill", tags=["teaching"]),
    ScriptInfo("teach_untruncate", ScriptCategory.LEARNING, "Teach untruncate", tags=["teaching"]),
    ScriptInfo("check_learning_status", ScriptCategory.LEARNING, "Check learning status", tags=["status"]),
    ScriptInfo("alignment_calibration_3cycles", ScriptCategory.LEARNING, "Alignment calibration 3 cycles", tags=["alignment"]),
    ScriptInfo("budget_forcing", ScriptCategory.LEARNING, "Budget forcing", tags=["budget"]),
]

# Genesis scripts
GENESIS_SCRIPTS = [
    ScriptInfo("run_birth_cycle", ScriptCategory.GENESIS, "Run birth cycle", tags=["genesis"]),
    ScriptInfo("run_genesis_historian", ScriptCategory.GENESIS, "Run genesis historian", tags=["genesis"]),
    ScriptInfo("trigger_crystallization", ScriptCategory.GENESIS, "Trigger crystallization", tags=["genesis"]),
    ScriptInfo("expand_global_mind", ScriptCategory.GENESIS, "Expand global mind", tags=["genesis"]),
    ScriptInfo("run_cycle_test", ScriptCategory.GENESIS, "Run cycle test", tags=["test"]),
]

# Manifold scripts
MANIFOLD_SCRIPTS = [
    ScriptInfo("activate_manifold", ScriptCategory.MANIFOLD, "Activate manifold", tags=["manifold"]),
    ScriptInfo("list_manifold_skills", ScriptCategory.MANIFOLD, "List manifold skills", tags=["manifold"]),
    ScriptInfo("start_topology_mapper", ScriptCategory.MANIFOLD, "Start topology mapper", tags=["topology"]),
]

# Gardener scripts
GARDENER_SCRIPTS = [
    ScriptInfo("filesystem_gardener_daemon", ScriptCategory.GARDENER, "Filesystem gardener daemon", tags=["gardener"]),
    ScriptInfo("sovereign_gardener_daemon", ScriptCategory.GARDENER, "Sovereign gardener daemon", tags=["gardener"]),
    ScriptInfo("gardener_summary", ScriptCategory.GARDENER, "Gardener summary", tags=["report"]),
    ScriptInfo("gardener_final_report", ScriptCategory.GARDENER, "Gardener final report", tags=["report"]),
]

# Miscellaneous scripts
MISC_SCRIPTS = [
    ScriptInfo("mcp_demo", ScriptCategory.MISC, "MCP demo", tags=["demo"]),
    ScriptInfo("run_gpia_server", ScriptCategory.MISC, "Run GPIA server", tags=["server"]),
    ScriptInfo("run_interview", ScriptCategory.MISC, "Run interview", tags=["interview"]),
    ScriptInfo("expectations_dialogue", ScriptCategory.MISC, "Expectations dialogue", tags=["dialogue"]),
    ScriptInfo("prospect", ScriptCategory.MISC, "Prospect script", tags=["misc"]),
    ScriptInfo("demonstrate_blindspot", ScriptCategory.MISC, "Demonstrate blindspot", tags=["demo"]),
    ScriptInfo("ch_cli", ScriptCategory.MISC, "CH CLI interface", tags=["cli"]),
    ScriptInfo("realtime_alignment_monitor", ScriptCategory.MISC, "Realtime alignment monitor", tags=["monitor"]),
    ScriptInfo("monitor_budget_system", ScriptCategory.MISC, "Monitor budget system", tags=["monitor"]),
    ScriptInfo("ollama_manifest_probe", ScriptCategory.MISC, "Ollama manifest probe", tags=["ollama"]),
    ScriptInfo("reconfigure_local_models", ScriptCategory.MISC, "Reconfigure local models", tags=["models"]),
    ScriptInfo("evaluate_gpia", ScriptCategory.MISC, "Evaluate GPIA", tags=["evaluation"]),
    ScriptInfo("run_gpia_eval", ScriptCategory.MISC, "Run GPIA evaluation", tags=["evaluation"]),
    ScriptInfo("count_skills", ScriptCategory.MISC, "Count skills", tags=["skills"]),
    ScriptInfo("gpi_identity_task", ScriptCategory.MISC, "GPI identity task", tags=["identity"]),
    ScriptInfo("adoption_coordinator", ScriptCategory.MISC, "Adoption coordinator", tags=["adoption"]),
    ScriptInfo("adversarial_smoke", ScriptCategory.MISC, "Adversarial smoke test", tags=["test"]),
    ScriptInfo("annex_enemy_command", ScriptCategory.MISC, "Annex command", tags=["command"]),
    ScriptInfo("flood_white_cells", ScriptCategory.MISC, "Flood white cells", tags=["cells"]),
]

# All scripts combined
ALL_SCRIPTS = (
    RESEARCH_SCRIPTS +
    ADMIN_SCRIPTS +
    AGENTS_SCRIPTS +
    VALIDATION_SCRIPTS +
    DATA_SCRIPTS +
    NETWORK_SCRIPTS +
    BUILD_SCRIPTS +
    LEARNING_SCRIPTS +
    GENESIS_SCRIPTS +
    MANIFOLD_SCRIPTS +
    GARDENER_SCRIPTS +
    MISC_SCRIPTS
)


# ============================================================================
# SCRIPT CATALOG
# ============================================================================

class ScriptCatalog:
    """
    Catalog of all utility scripts.

    Provides:
    - Lookup by name, category, tag
    - Statistics
    - Export to JSON
    """

    def __init__(self, scripts: Optional[List[ScriptInfo]] = None):
        self._scripts = {s.name: s for s in (scripts or ALL_SCRIPTS)}
        self._by_category: Dict[ScriptCategory, List[ScriptInfo]] = {}
        self._by_tag: Dict[str, List[ScriptInfo]] = {}
        self._index()

    def _index(self) -> None:
        """Build indexes."""
        for script in self._scripts.values():
            # By category
            if script.category not in self._by_category:
                self._by_category[script.category] = []
            self._by_category[script.category].append(script)

            # By tag
            for tag in script.tags:
                if tag not in self._by_tag:
                    self._by_tag[tag] = []
                self._by_tag[tag].append(script)

    def get(self, name: str) -> Optional[ScriptInfo]:
        """Get script by name."""
        return self._scripts.get(name)

    def list_all(self) -> List[ScriptInfo]:
        """List all scripts."""
        return list(self._scripts.values())

    def list_by_category(self, category: ScriptCategory) -> List[ScriptInfo]:
        """List scripts by category."""
        return self._by_category.get(category, [])

    def list_by_tag(self, tag: str) -> List[ScriptInfo]:
        """List scripts by tag."""
        return self._by_tag.get(tag, [])

    def search(self, query: str) -> List[ScriptInfo]:
        """Search scripts by name or description."""
        query = query.lower()
        return [
            s for s in self._scripts.values()
            if query in s.name.lower() or query in s.description.lower()
        ]

    def categories(self) -> List[ScriptCategory]:
        """List all categories with scripts."""
        return list(self._by_category.keys())

    def tags(self) -> List[str]:
        """List all tags."""
        return list(self._by_tag.keys())

    def stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return {
            "total": len(self._scripts),
            "categories": {cat.value: len(scripts) for cat, scripts in self._by_category.items()},
            "top_tags": sorted(
                [(tag, len(scripts)) for tag, scripts in self._by_tag.items()],
                key=lambda x: -x[1]
            )[:10],
        }


# ============================================================================
# SCRIPT RUNNER
# ============================================================================

@dataclass
class ScriptResult:
    """Result of script execution."""
    script: str
    status: ScriptStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    timestamp: str = ""

    def success(self) -> bool:
        """Check if script succeeded."""
        return self.status == ScriptStatus.SUCCESS


class ScriptRunner:
    """
    Safe script execution with logging and timeout.
    """

    def __init__(
        self,
        scripts_dir: Optional[str] = None,
        python_path: str = sys.executable,
        timeout: int = 300
    ):
        self.scripts_dir = Path(scripts_dir) if scripts_dir else None
        self.python_path = python_path
        self.timeout = timeout
        self._history: List[ScriptResult] = []

    def run(
        self,
        script_name: str,
        args: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> ScriptResult:
        """
        Run a script by name.

        Args:
            script_name: Script name (without .py)
            args: Command-line arguments
            timeout: Optional timeout override

        Returns:
            ScriptResult with execution details
        """
        import time

        start = time.time()

        # Find script path
        if self.scripts_dir:
            script_path = self.scripts_dir / f"{script_name}.py"
        else:
            script_path = Path(f"{script_name}.py")

        if not script_path.exists():
            return ScriptResult(
                script=script_name,
                status=ScriptStatus.FAILED,
                stderr=f"Script not found: {script_path}",
                timestamp=datetime.now().isoformat()
            )

        # Build command
        cmd = [self.python_path, str(script_path)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout
            )

            duration = (time.time() - start) * 1000

            script_result = ScriptResult(
                script=script_name,
                status=ScriptStatus.SUCCESS if result.returncode == 0 else ScriptStatus.FAILED,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration_ms=duration,
                timestamp=datetime.now().isoformat()
            )

        except subprocess.TimeoutExpired:
            script_result = ScriptResult(
                script=script_name,
                status=ScriptStatus.FAILED,
                stderr=f"Script timed out after {timeout or self.timeout}s",
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            script_result = ScriptResult(
                script=script_name,
                status=ScriptStatus.FAILED,
                stderr=str(e),
                timestamp=datetime.now().isoformat()
            )

        self._history.append(script_result)
        return script_result

    def dry_run(self, script_name: str) -> ScriptResult:
        """Simulate running a script without execution."""
        return ScriptResult(
            script=script_name,
            status=ScriptStatus.SKIPPED,
            stdout=f"[DRY RUN] Would execute: {script_name}",
            timestamp=datetime.now().isoformat()
        )

    def history(self, limit: int = 10) -> List[ScriptResult]:
        """Get execution history."""
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()


# ============================================================================
# SCRIPT MANAGER (Main Interface)
# ============================================================================

class ScriptManager:
    """
    Main interface for script management.

    Combines catalog and runner functionality.
    """

    def __init__(self, scripts_dir: Optional[str] = None):
        self.catalog = ScriptCatalog()
        self.runner = ScriptRunner(scripts_dir)

    def get(self, name: str) -> Optional[ScriptInfo]:
        """Get script info by name."""
        return self.catalog.get(name)

    def list_category(self, category: ScriptCategory) -> List[ScriptInfo]:
        """List scripts in category."""
        return self.catalog.list_by_category(category)

    def search(self, query: str) -> List[ScriptInfo]:
        """Search scripts."""
        return self.catalog.search(query)

    def run(self, name: str, args: Optional[List[str]] = None) -> ScriptResult:
        """Run a script."""
        return self.runner.run(name, args)

    def stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return self.catalog.stats()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_script_catalog() -> ScriptCatalog:
    """Create a script catalog."""
    return ScriptCatalog()


def create_script_runner(scripts_dir: Optional[str] = None) -> ScriptRunner:
    """Create a script runner."""
    return ScriptRunner(scripts_dir)


def create_script_manager(scripts_dir: Optional[str] = None) -> ScriptManager:
    """Create a script manager."""
    return ScriptManager(scripts_dir)


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Scripts Extension...")

    # Test catalog
    catalog = ScriptCatalog()
    stats = catalog.stats()
    print(f"Total scripts: {stats['total']}")
    print(f"Categories: {len(stats['categories'])}")

    # Test search
    results = catalog.search("bsd")
    print(f"BSD scripts: {len(results)}")

    # Test by category
    research = catalog.list_by_category(ScriptCategory.RESEARCH)
    print(f"Research scripts: {len(research)}")

    # Test by tag
    riemann = catalog.list_by_tag("riemann")
    print(f"Riemann scripts: {len(riemann)}")

    print("\nAll tests passed!")
