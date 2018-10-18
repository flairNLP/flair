import pytest

@pytest.fixture(scope="module")
def tasks_base_path():
    return Path(__file__).parent / 'resources' / 'tasks'
