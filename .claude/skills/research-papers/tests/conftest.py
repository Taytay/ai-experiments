"""Pytest configuration for research-papers skill tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires network)"
    )


def pytest_addoption(parser):
    """Add --integration command line option."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires network; arXiv/Semantic Scholar rate limits apply)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration is passed."""
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="Need --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
