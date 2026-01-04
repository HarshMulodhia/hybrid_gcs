"""
Pytest configuration for Hybrid-GCS tests.

Provides common fixtures and configuration for all test suites.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Configure pytest."""
    # Set random seed for reproducible tests
    np.random.seed(42)


@pytest.fixture(scope="session")
def random_seed():
    """Provide random seed for tests."""
    return 42


@pytest.fixture
def reset_random():
    """Reset random seed before each test."""
    np.random.seed(42)
    yield
    np.random.seed(None)
