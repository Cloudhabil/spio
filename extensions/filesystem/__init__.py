"""
Filesystem Extension - VFS and File Organization

Components:
1. FilesystemGardener - Autonomous file organization
2. ArtifactClassifier - File type classification
3. VirtualFS - Virtual filesystem abstraction
4. FileWatcher - Directory monitoring
5. OrganizationLedger - Movement audit trail

Reference: CLI-main/src/core/filesystem_gardener.py
"""

from .filesystem_core import (
    # Enums
    ArtifactType,
    FileOperation,

    # Data classes
    FileArtifact,
    OrganizationAction,
    WatchEvent,

    # VFS
    VirtualFS,
    VFSNode,
    VFSFile,
    VFSDirectory,

    # Gardener
    FilesystemGardener,
    ArtifactClassifier,
    OrganizationLedger,

    # Watcher
    FileWatcher,

    # Main interface
    Filesystem,

    # Factories
    create_filesystem,
    create_gardener,
    get_filesystem,
)

__all__ = [
    "ArtifactType", "FileOperation",
    "FileArtifact", "OrganizationAction", "WatchEvent",
    "VirtualFS", "VFSNode", "VFSFile", "VFSDirectory",
    "FilesystemGardener", "ArtifactClassifier", "OrganizationLedger",
    "FileWatcher",
    "Filesystem",
    "create_filesystem", "create_gardener", "get_filesystem",
]
