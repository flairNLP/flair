import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def resources_path():
    return Path(__file__).parent / 'resources'


@pytest.fixture(scope="module")
def tasks_base_path(resources_path):
    return resources_path / 'tasks'
