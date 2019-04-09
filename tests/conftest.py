import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def resources_path():
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def tasks_base_path(resources_path):
    return resources_path / "tasks"


@pytest.fixture(scope="module")
def results_base_path(resources_path):
    return resources_path / "results"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") and config.getoption("--runintegration"):
        return

    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(
            reason="need --runintegration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
