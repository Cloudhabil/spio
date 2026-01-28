"""
Filesystem Core - VFS, Gardener, and File Organization

Based on: CLI-main/src/core/filesystem_gardener.py
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ArtifactType(Enum):
    """File artifact classifications."""
    SKILL_SYNTHESIZED = "skill_synthesized"
    EVAL_BENCHMARK = "eval_benchmark"
    EXPERIMENT_ACTIVE = "experiment_active"
    DATA_LEDGER = "data_ledger"
    CONFIG = "config"
    LOG = "log"
    CACHE = "cache"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


class FileOperation(Enum):
    """File operation types."""
    CREATE = auto()
    MODIFY = auto()
    DELETE = auto()
    MOVE = auto()
    COPY = auto()
    RENAME = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FileArtifact:
    """Metadata for a file artifact."""
    path: Path
    artifact_type: ArtifactType = ArtifactType.UNKNOWN
    size_bytes: int = 0
    hash_sha256: str = ""
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrganizationAction:
    """Record of a file organization action."""
    action_id: str
    source: Path
    destination: Path
    artifact_type: ArtifactType
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None


@dataclass
class WatchEvent:
    """File system watch event."""
    path: Path
    operation: FileOperation
    timestamp: float = field(default_factory=time.time)
    is_directory: bool = False


# =============================================================================
# VFS NODES
# =============================================================================

@dataclass
class VFSNode:
    """Base class for VFS nodes."""
    name: str
    parent: Optional["VFSDirectory"] = None
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    permissions: int = 0o644

    @property
    def path(self) -> str:
        if self.parent is None:
            return "/" + self.name
        return self.parent.path + "/" + self.name


@dataclass
class VFSFile(VFSNode):
    """Virtual file."""
    content: bytes = b""
    size: int = 0

    def read(self) -> bytes:
        return self.content

    def write(self, data: bytes):
        self.content = data
        self.size = len(data)
        self.modified_at = time.time()


@dataclass
class VFSDirectory(VFSNode):
    """Virtual directory."""
    children: Dict[str, VFSNode] = field(default_factory=dict)
    permissions: int = 0o755

    def add(self, node: VFSNode):
        node.parent = self
        self.children[node.name] = node
        self.modified_at = time.time()

    def remove(self, name: str) -> Optional[VFSNode]:
        return self.children.pop(name, None)

    def get(self, name: str) -> Optional[VFSNode]:
        return self.children.get(name)

    def list(self) -> List[str]:
        return list(self.children.keys())


# =============================================================================
# VIRTUAL FILESYSTEM
# =============================================================================

class VirtualFS:
    """
    Virtual filesystem abstraction.

    Provides a POSIX-like interface over an in-memory or backed filesystem.
    """

    def __init__(self):
        self.root = VFSDirectory(name="")
        self._lock = threading.Lock()

    def _resolve(self, path: str) -> Optional[VFSNode]:
        """Resolve a path to a node."""
        if path == "/" or path == "":
            return self.root

        parts = path.strip("/").split("/")
        node = self.root

        for part in parts:
            if not isinstance(node, VFSDirectory):
                return None
            node = node.get(part)
            if node is None:
                return None

        return node

    def exists(self, path: str) -> bool:
        with self._lock:
            return self._resolve(path) is not None

    def is_file(self, path: str) -> bool:
        with self._lock:
            node = self._resolve(path)
            return isinstance(node, VFSFile)

    def is_dir(self, path: str) -> bool:
        with self._lock:
            node = self._resolve(path)
            return isinstance(node, VFSDirectory)

    def mkdir(self, path: str, parents: bool = False) -> bool:
        with self._lock:
            parts = path.strip("/").split("/")
            node = self.root

            for i, part in enumerate(parts):
                if not isinstance(node, VFSDirectory):
                    return False

                child = node.get(part)
                if child is None:
                    if parents or i == len(parts) - 1:
                        new_dir = VFSDirectory(name=part)
                        node.add(new_dir)
                        node = new_dir
                    else:
                        return False
                elif isinstance(child, VFSDirectory):
                    node = child
                else:
                    return False

            return True

    def create_file(self, path: str, content: bytes = b"") -> bool:
        with self._lock:
            parent_path = "/".join(path.strip("/").split("/")[:-1])
            name = path.strip("/").split("/")[-1]

            parent = self._resolve("/" + parent_path)
            if not isinstance(parent, VFSDirectory):
                return False

            file = VFSFile(name=name, content=content, size=len(content))
            parent.add(file)
            return True

    def read_file(self, path: str) -> Optional[bytes]:
        with self._lock:
            node = self._resolve(path)
            if isinstance(node, VFSFile):
                return node.read()
            return None

    def write_file(self, path: str, content: bytes) -> bool:
        with self._lock:
            node = self._resolve(path)
            if isinstance(node, VFSFile):
                node.write(content)
                return True
            return False

    def delete(self, path: str) -> bool:
        with self._lock:
            parts = path.strip("/").split("/")
            if len(parts) == 0:
                return False

            parent_path = "/" + "/".join(parts[:-1])
            name = parts[-1]

            parent = self._resolve(parent_path)
            if isinstance(parent, VFSDirectory):
                return parent.remove(name) is not None
            return False

    def list_dir(self, path: str) -> List[str]:
        with self._lock:
            node = self._resolve(path)
            if isinstance(node, VFSDirectory):
                return node.list()
            return []


# =============================================================================
# ARTIFACT CLASSIFIER
# =============================================================================

class ArtifactClassifier:
    """Classifies files by type based on patterns."""

    PATTERNS = {
        ArtifactType.SKILL_SYNTHESIZED: ["skill_*.py", "synthesized_*.py"],
        ArtifactType.EVAL_BENCHMARK: ["benchmark_*.json", "eval_*.json"],
        ArtifactType.DATA_LEDGER: ["ledger_*.jsonl", "*.ledger"],
        ArtifactType.CONFIG: ["*.yaml", "*.yml", "*.toml", "config.json"],
        ArtifactType.LOG: ["*.log", "logs/*"],
        ArtifactType.CACHE: ["*.cache", ".cache/*", "__pycache__/*"],
        ArtifactType.MODEL: ["*.pt", "*.pth", "*.onnx", "*.bin"],
        ArtifactType.CHECKPOINT: ["checkpoint_*.pt", "*.ckpt"],
        ArtifactType.DOCUMENTATION: ["*.md", "*.rst", "docs/*"],
    }

    def classify(self, path: Path) -> ArtifactType:
        """Classify a file by its path."""
        name = path.name.lower()
        path_str = str(path).lower()

        for artifact_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if self._matches(path_str, name, pattern):
                    return artifact_type

        return ArtifactType.UNKNOWN

    def _matches(self, path_str: str, name: str, pattern: str) -> bool:
        """Check if path matches pattern."""
        import fnmatch
        if "/" in pattern:
            return fnmatch.fnmatch(path_str, f"*{pattern}")
        return fnmatch.fnmatch(name, pattern)


# =============================================================================
# ORGANIZATION LEDGER
# =============================================================================

class OrganizationLedger:
    """Audit trail for file organization actions."""

    def __init__(self, ledger_path: Path = None):
        self.ledger_path = ledger_path or Path("organization.ledger")
        self._actions: List[OrganizationAction] = []
        self._lock = threading.Lock()

    def record(self, action: OrganizationAction):
        """Record an action."""
        with self._lock:
            self._actions.append(action)
            self._persist(action)

    def _persist(self, action: OrganizationAction):
        """Persist action to ledger file."""
        try:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps({
                    "id": action.action_id,
                    "source": str(action.source),
                    "destination": str(action.destination),
                    "type": action.artifact_type.value,
                    "timestamp": action.timestamp,
                    "success": action.success,
                    "error": action.error,
                }) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist action: {e}")

    def get_history(self, limit: int = 100) -> List[OrganizationAction]:
        """Get recent actions."""
        with self._lock:
            return self._actions[-limit:]


# =============================================================================
# FILE WATCHER
# =============================================================================

class FileWatcher:
    """Watches directories for changes."""

    def __init__(self, paths: List[Path] = None):
        self.paths = paths or []
        self._callbacks: List[Callable[[WatchEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._known_files: Dict[Path, float] = {}

    def add_callback(self, callback: Callable[[WatchEvent], None]):
        """Add event callback."""
        self._callbacks.append(callback)

    def start(self):
        """Start watching."""
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _watch_loop(self):
        """Main watch loop (polling-based)."""
        while self._running:
            for path in self.paths:
                if path.is_dir():
                    self._scan_directory(path)
            time.sleep(1.0)

    def _scan_directory(self, directory: Path):
        """Scan directory for changes."""
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    mtime = item.stat().st_mtime
                    if item not in self._known_files:
                        self._emit(WatchEvent(item, FileOperation.CREATE))
                    elif self._known_files[item] != mtime:
                        self._emit(WatchEvent(item, FileOperation.MODIFY))
                    self._known_files[item] = mtime
        except Exception as e:
            logger.debug(f"Scan error: {e}")

    def _emit(self, event: WatchEvent):
        """Emit event to callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Callback error: {e}")


# =============================================================================
# FILESYSTEM GARDENER
# =============================================================================

class FilesystemGardener:
    """
    Autonomous file organization agent.

    Features:
    - Watches for new files
    - Classifies artifacts
    - Organizes into appropriate directories
    - Maintains audit trail (never deletes)
    """

    def __init__(self, root: Path, auto_organize: bool = False):
        self.root = Path(root)
        self.auto_organize = auto_organize
        self.classifier = ArtifactClassifier()
        self.ledger = OrganizationLedger(self.root / ".organization.ledger")
        self.watcher = FileWatcher([self.root])
        self._action_counter = 0

        if auto_organize:
            self.watcher.add_callback(self._on_file_event)

    def start(self):
        """Start the gardener."""
        self.watcher.start()
        logger.info(f"FilesystemGardener started watching: {self.root}")

    def stop(self):
        """Stop the gardener."""
        self.watcher.stop()
        logger.info("FilesystemGardener stopped")

    def classify(self, path: Path) -> FileArtifact:
        """Classify a file."""
        artifact_type = self.classifier.classify(path)
        stat = path.stat() if path.exists() else None

        return FileArtifact(
            path=path,
            artifact_type=artifact_type,
            size_bytes=stat.st_size if stat else 0,
            hash_sha256=self._compute_hash(path) if path.exists() else "",
            created_at=stat.st_ctime if stat else time.time(),
            modified_at=stat.st_mtime if stat else time.time(),
        )

    def organize(self, path: Path) -> Optional[OrganizationAction]:
        """Organize a file into appropriate directory."""
        artifact = self.classify(path)
        if artifact.artifact_type == ArtifactType.UNKNOWN:
            return None

        dest_dir = self.root / artifact.artifact_type.value
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / path.name

        self._action_counter += 1
        action = OrganizationAction(
            action_id=f"org-{self._action_counter}",
            source=path,
            destination=dest_path,
            artifact_type=artifact.artifact_type,
        )

        try:
            if not dest_path.exists():
                path.rename(dest_path)
                action.success = True
        except Exception as e:
            action.success = False
            action.error = str(e)

        self.ledger.record(action)
        return action

    def _on_file_event(self, event: WatchEvent):
        """Handle file events."""
        if event.operation == FileOperation.CREATE:
            self.organize(event.path)

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""


# =============================================================================
# FILESYSTEM (UNIFIED INTERFACE)
# =============================================================================

class Filesystem:
    """Unified filesystem interface."""

    def __init__(self, root: Path = None):
        self.root = Path(root) if root else Path(".")
        self.vfs = VirtualFS()
        self.gardener = FilesystemGardener(self.root)
        self.classifier = ArtifactClassifier()

    def start(self):
        self.gardener.start()

    def stop(self):
        self.gardener.stop()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_filesystem: Optional[Filesystem] = None


def create_filesystem(root: Path = None) -> Filesystem:
    return Filesystem(root)


def create_gardener(root: Path, auto_organize: bool = False) -> FilesystemGardener:
    return FilesystemGardener(root, auto_organize)


def get_filesystem() -> Filesystem:
    global _filesystem
    if _filesystem is None:
        _filesystem = Filesystem()
    return _filesystem
