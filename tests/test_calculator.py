"""
Tests for Brahim's Calculator

Verifies the deterministic computation functions.
"""

import pytest
from math import pi, isclose

from sovereign_pio.constants import PHI, OMEGA
from sovereign_pio.calculator import D, Theta, Energy, x_from_D, lucas


class TestConstants:
    """Test Brahim's Calculator constants."""

    def test_phi_value(self):
        """PHI should be the golden ratio."""
        expected = 1.6180339887498949
        assert isclose(PHI, expected, rel_tol=1e-15)

    def test_omega_inverse(self):
        """OMEGA should be 1/PHI."""
        assert isclose(OMEGA, 1 / PHI, rel_tol=1e-15)

    def test_phi_omega_relationship(self):
        """PHI * OMEGA should equal 1."""
        assert isclose(PHI * OMEGA, 1.0, rel_tol=1e-15)


class TestDimensionFunction:
    """Test the D(x) dimension function."""

    def test_d_of_phi(self):
        """D(PHI) should be -1."""
        assert isclose(D(PHI), -1.0, rel_tol=1e-10)

    def test_d_of_omega(self):
        """D(1/PHI) should be 1."""
        assert isclose(D(OMEGA), 1.0, rel_tol=1e-10)

    def test_d_of_one(self):
        """D(1) should be 0."""
        assert isclose(D(1.0), 0.0, abs_tol=1e-15)

    def test_d_requires_positive(self):
        """D(x) should raise for x <= 0."""
        with pytest.raises(ValueError):
            D(0)
        with pytest.raises(ValueError):
            D(-1)


class TestThetaFunction:
    """Test the Theta(x) phase function."""

    def test_theta_of_one(self):
        """Theta(1) should be 2*PI."""
        assert isclose(Theta(1.0), 2 * pi, rel_tol=1e-15)

    def test_theta_of_half(self):
        """Theta(0.5) should be PI."""
        assert isclose(Theta(0.5), pi, rel_tol=1e-15)

    def test_theta_linear(self):
        """Theta should be linear."""
        assert isclose(Theta(2.0), 2 * Theta(1.0), rel_tol=1e-15)


class TestEnergyFunction:
    """Test the Energy function."""

    def test_energy_conservation(self):
        """Energy should always be 2*PI."""
        test_values = [0.1, 0.5, 1.0, PHI, 2.0, 10.0]
        for x in test_values:
            assert isclose(Energy(x), 2 * pi, rel_tol=1e-10), f"Energy({x}) != 2*PI"

    def test_energy_requires_positive(self):
        """Energy(x) should raise for x <= 0."""
        with pytest.raises(ValueError):
            Energy(0)


class TestInverseFunction:
    """Test the x_from_D inverse function."""

    def test_inverse_of_one(self):
        """x_from_D(1) should be OMEGA."""
        assert isclose(x_from_D(1.0), OMEGA, rel_tol=1e-15)

    def test_inverse_of_zero(self):
        """x_from_D(0) should be 1."""
        assert isclose(x_from_D(0.0), 1.0, rel_tol=1e-15)

    def test_roundtrip(self):
        """D and x_from_D should be inverses."""
        test_values = [0.1, 0.5, 1.0, PHI, 2.0]
        for x in test_values:
            d = D(x)
            recovered = x_from_D(d)
            assert isclose(recovered, x, rel_tol=1e-10), f"Roundtrip failed for {x}"


class TestLucasNumbers:
    """Test Lucas number computation."""

    def test_first_twelve(self):
        """First 12 Lucas numbers for dimensions."""
        expected = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
        for i, exp in enumerate(expected, 1):
            assert lucas(i) == exp, f"lucas({i}) should be {exp}"

    def test_total_states(self):
        """Total states should be 840."""
        total = sum(lucas(i) for i in range(1, 13))
        assert total == 840

    def test_lucas_requires_positive(self):
        """lucas(n) should raise for n < 1."""
        with pytest.raises(ValueError):
            lucas(0)
