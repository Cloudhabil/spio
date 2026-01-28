"""
Tests for Sovereign PIO Security Module

Verifies security controls and sandbox enforcement.
"""

import pytest
from pathlib import Path

from security import (
    SecurityViolation,
    SandboxViolation,
    CommandValidator,
    FilesystemGuard,
)


class TestCommandValidator:
    """Test command validation security."""

    @pytest.fixture
    def validator(self):
        return CommandValidator()

    def test_safe_command(self, validator):
        """Safe commands should pass."""
        assert validator.validate("ls -la")
        assert validator.validate("echo hello")
        assert validator.validate("cat file.txt")

    def test_blocked_binary_bash(self, validator):
        """Bash should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("bash -c 'echo test'")

    def test_blocked_binary_nc(self, validator):
        """Netcat should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("nc -l 8080")

    def test_blocked_binary_dd(self, validator):
        """dd should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("dd if=/dev/zero of=/dev/sda")

    def test_dangerous_rm(self, validator):
        """rm -rf should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("rm -rf /")
        with pytest.raises(SecurityViolation):
            validator.validate("rm -fr /home")

    def test_path_traversal(self, validator):
        """Path traversal should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("cat ../../../etc/passwd")

    def test_command_substitution(self, validator):
        """Command substitution should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("echo $(whoami)")
        with pytest.raises(SecurityViolation):
            validator.validate("echo `id`")

    def test_sensitive_files(self, validator):
        """Access to sensitive files should be blocked."""
        with pytest.raises(SecurityViolation):
            validator.validate("cat /etc/shadow")
        with pytest.raises(SecurityViolation):
            validator.validate("cat /etc/passwd")


class TestFilesystemGuard:
    """Test filesystem sandboxing."""

    @pytest.fixture
    def guard(self, tmp_path):
        # Create test directory structure
        (tmp_path / "src").mkdir()
        (tmp_path / "config").mkdir()
        (tmp_path / "data").mkdir()
        (tmp_path / "logs").mkdir()
        (tmp_path / "temp").mkdir()
        (tmp_path / "docs").mkdir()
        (tmp_path / "README.md").touch()

        return FilesystemGuard(project_root=tmp_path)

    def test_read_allowed_in_src(self, guard):
        """Reading from src/ should be allowed."""
        assert guard.can_read("src/file.py")

    def test_read_allowed_readme(self, guard):
        """Reading README.md should be allowed."""
        assert guard.can_read("README.md")

    def test_write_allowed_in_data(self, guard):
        """Writing to data/ should be allowed."""
        assert guard.can_write("data/output.json")

    def test_write_allowed_in_logs(self, guard):
        """Writing to logs/ should be allowed."""
        assert guard.can_write("logs/app.log")

    def test_write_blocked_in_src(self, guard):
        """Writing to src/ should be blocked."""
        assert not guard.can_write("src/file.py")

    def test_path_traversal_blocked(self, guard):
        """Path traversal should be blocked."""
        with pytest.raises(SandboxViolation):
            guard._resolve_path("../../../etc/passwd")

    def test_validate_read_raises(self, guard):
        """validate_read should raise for blocked paths."""
        with pytest.raises(SandboxViolation):
            guard.validate_read("secret/passwords.txt")

    def test_validate_write_raises(self, guard):
        """validate_write should raise for blocked paths."""
        with pytest.raises(SandboxViolation):
            guard.validate_write("src/malicious.py")
