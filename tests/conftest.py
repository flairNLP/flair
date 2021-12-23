from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def resources_path():
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def tasks_base_path(resources_path):
    return resources_path / "tasks"


@pytest.fixture
def results_base_path(resources_path):
    path = resources_path / "results"
    try:
        yield path
    finally:
        for p in reversed(list(path.rglob("*"))):
            if p.is_file():
                p.unlink()
            else:
                p.rmdir()
        if path.is_dir():
            path.rmdir()


def pytest_addoption(parser):
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
