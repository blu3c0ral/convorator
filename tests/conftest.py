# tests/conftest.py
import pytest


@pytest.fixture
def sample_data():
    return {"key": "value"}


# Hook to modify test collection
def pytest_collection_modifyitems(items):
    for item in items:
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
