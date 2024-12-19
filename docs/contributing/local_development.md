# Local Development

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Most classes and methods are documented, so finding your way around
the code should hopefully be easy.

## Setup

Flair requires python-3.9 or higher. To make sure our code also runs on the oldest supported
python version, it is recommended to use python-3.9.x for flair development.

Create a python environment of your preference and run:
```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Tests

To only run typechecks and check the code formatting execute:

```bash
pytest flair
```

To run all basic tests execute:

```bash
pytest
```

To run integration tests execute:

```bash
pytest --runintegration
```

The integration tests will train small models and therefore take more time.
In general, it is recommended to ensure all basic tests are running through before testing the integration tests

## Code Formatting

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black) and for standardizing imports we use [ruff](https://github.com/charliermarsh/ruff).
If your code is not formatted properly, the tests will fail.

We recommend configuring your IDE to run these formatters for you, but you can also always run them manually via
`black . && ruff --fix .` in the flair root folder.