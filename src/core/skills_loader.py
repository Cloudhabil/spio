"""
Skill Discovery & Metadata Loading

YAML-based skill metadata parser for lazy skill loading.
Discovers skills without importing code, enabling efficient skill management.

Features:
- Pattern-based skill discovery (*/skill.yaml)
- Metadata extraction (id, name, version, permissions)
- Input/output schema loading
- Deferred code import
"""

import yaml
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum


class SkillCategory(Enum):
    """Skill categories for organization."""
    CORE = "core"
    AUTOMATION = "automation"
    COMMUNICATION = "communication"
    COMPUTATION = "computation"
    CREATIVITY = "creativity"
    DATA = "data"
    INTEGRATION = "integration"
    SECURITY = "security"
    SYSTEM = "system"
    CUSTOM = "custom"


class Permission(Enum):
    """Permissions a skill may require."""
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    NETWORK = "network"
    EXECUTE = "execute"
    MEMORY = "memory"
    GPU = "gpu"
    SECRETS = "secrets"
    ADMIN = "admin"


@dataclass
class SkillSchema:
    """Input/output schema for a skill."""
    type: str = "object"
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillSchema":
        return cls(
            type=data.get("type", "object"),
            properties=data.get("properties", {}),
            required=data.get("required", []),
        )

    def validate(self, data: dict) -> tuple[bool, Optional[str]]:
        """Basic validation against schema."""
        # Check required fields
        for req in self.required:
            if req not in data:
                return False, f"Missing required field: {req}"

        # Type checking for properties
        for key, value in data.items():
            if key in self.properties:
                prop_schema = self.properties[key]
                expected_type = prop_schema.get("type")
                if expected_type:
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }
                    if expected_type in type_map:
                        if not isinstance(value, type_map[expected_type]):
                            return False, f"Field {key} should be {expected_type}"

        return True, None


@dataclass
class SkillMetadata:
    """Metadata for a skill without loading code."""
    id: str
    name: str
    version: str
    description: str
    category: SkillCategory = SkillCategory.CUSTOM
    permissions: List[Permission] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    input_schema: Optional[SkillSchema] = None
    output_schema: Optional[SkillSchema] = None

    # Source information
    path: Optional[Path] = None
    module: Optional[str] = None
    entry_point: str = "execute"

    # Runtime state
    loaded: bool = False
    _executor: Optional[Callable] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category.value,
            "permissions": [p.value for p in self.permissions],
            "capabilities": self.capabilities,
            "loaded": self.loaded,
        }


class SkillLoader:
    """
    Skill Discovery and Loading System.

    Discovers skills from filesystem patterns and loads metadata
    without importing code until needed.

    Example:
        loader = SkillLoader("./skills")
        loader.discover()

        # List available skills
        for skill in loader.list():
            print(f"{skill.name}: {skill.description}")

        # Load and execute
        result = await loader.execute("my-skill", {"input": "data"})
    """

    SKILL_MANIFEST = "skill.yaml"

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, SkillMetadata] = {}
        self._discovered = False

    def discover(self, force: bool = False) -> int:
        """
        Discover skills in the skills directory.

        Returns number of skills discovered.
        """
        if self._discovered and not force:
            return len(self.skills)

        self.skills.clear()

        if not self.skills_dir.exists():
            self._discovered = True
            return 0

        # Find all skill.yaml files
        patterns = [
            self.skills_dir / "*" / self.SKILL_MANIFEST,
            self.skills_dir / "*" / "*" / self.SKILL_MANIFEST,
        ]

        for pattern in patterns:
            for manifest_path in self.skills_dir.glob(
                str(pattern.relative_to(self.skills_dir))
            ):
                try:
                    metadata = self._load_manifest(manifest_path)
                    if metadata:
                        self.skills[metadata.id] = metadata
                except Exception as e:
                    print(f"Error loading {manifest_path}: {e}")

        self._discovered = True
        return len(self.skills)

    def _load_manifest(self, path: Path) -> Optional[SkillMetadata]:
        """Load skill metadata from manifest file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "id" not in data:
            return None

        # Parse category
        category = SkillCategory.CUSTOM
        if "category" in data:
            try:
                category = SkillCategory(data["category"])
            except ValueError:
                pass

        # Parse permissions
        permissions = []
        for perm in data.get("permissions", []):
            try:
                permissions.append(Permission(perm))
            except ValueError:
                pass

        # Parse schemas
        input_schema = None
        if "input" in data:
            input_schema = SkillSchema.from_dict(data["input"])

        output_schema = None
        if "output" in data:
            output_schema = SkillSchema.from_dict(data["output"])

        return SkillMetadata(
            id=data["id"],
            name=data.get("name", data["id"]),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            category=category,
            permissions=permissions,
            capabilities=data.get("capabilities", []),
            input_schema=input_schema,
            output_schema=output_schema,
            path=path.parent,
            module=data.get("module"),
            entry_point=data.get("entry_point", "execute"),
        )

    def get(self, skill_id: str) -> Optional[SkillMetadata]:
        """Get skill metadata by ID."""
        if not self._discovered:
            self.discover()
        return self.skills.get(skill_id)

    def list(
        self,
        category: Optional[SkillCategory] = None,
        capability: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """
        List available skills with optional filtering.

        Args:
            category: Filter by category
            capability: Filter by capability

        Returns:
            List of matching skill metadata
        """
        if not self._discovered:
            self.discover()

        result = list(self.skills.values())

        if category:
            result = [s for s in result if s.category == category]

        if capability:
            result = [s for s in result if capability in s.capabilities]

        return result

    def search(self, query: str) -> List[SkillMetadata]:
        """Search skills by name or description."""
        if not self._discovered:
            self.discover()

        query_lower = query.lower()
        results = []

        for skill in self.skills.values():
            if (
                query_lower in skill.name.lower() or
                query_lower in skill.description.lower() or
                any(query_lower in cap.lower() for cap in skill.capabilities)
            ):
                results.append(skill)

        return results

    def load(self, skill_id: str) -> bool:
        """
        Load a skill's code (deferred import).

        Returns True if successful.
        """
        skill = self.get(skill_id)
        if not skill:
            return False

        if skill.loaded:
            return True

        try:
            # Determine module path
            if skill.module:
                module_path = skill.module
            else:
                # Default: look for main.py in skill directory
                module_path = str(skill.path / "main.py")

            if skill.path and (skill.path / "main.py").exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"skill_{skill.id}",
                    skill.path / "main.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Get entry point
                    if hasattr(module, skill.entry_point):
                        skill._executor = getattr(module, skill.entry_point)
                        skill.loaded = True
                        return True

            return False

        except Exception as e:
            print(f"Error loading skill {skill_id}: {e}")
            return False

    async def execute(
        self,
        skill_id: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a skill.

        Args:
            skill_id: The skill to execute
            inputs: Input data
            context: Optional execution context

        Returns:
            Skill output
        """
        skill = self.get(skill_id)
        if not skill:
            return {"error": f"Skill not found: {skill_id}"}

        # Validate inputs
        if skill.input_schema:
            valid, error = skill.input_schema.validate(inputs)
            if not valid:
                return {"error": f"Invalid input: {error}"}

        # Load if needed
        if not skill.loaded:
            if not self.load(skill_id):
                return {"error": f"Failed to load skill: {skill_id}"}

        if not skill._executor:
            return {"error": f"Skill has no executor: {skill_id}"}

        # Execute
        try:
            import asyncio
            if asyncio.iscoroutinefunction(skill._executor):
                result = await skill._executor(inputs, context)
            else:
                result = skill._executor(inputs, context)

            # Validate output
            if skill.output_schema and isinstance(result, dict):
                valid, error = skill.output_schema.validate(result)
                if not valid:
                    return {"error": f"Invalid output: {error}", "data": result}

            return {"success": True, "data": result}

        except Exception as e:
            return {"error": str(e)}

    def stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        if not self._discovered:
            self.discover()

        categories = {}
        for skill in self.skills.values():
            cat = skill.category.value
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_skills": len(self.skills),
            "loaded_skills": sum(1 for s in self.skills.values() if s.loaded),
            "categories": categories,
            "skills_dir": str(self.skills_dir),
        }
