# Contributing to Flair

We are happy to accept your contributions to make `flair` better and more awesome! To avoid unnecessary work on either
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/flairNLP/flair/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the
   commit guidelines below.


## Git Commit Guidelines

If there is already a ticket, use this number at the start of your commit message.
Use meaningful commit messages that described what you did.

**Example:** `GH-42: Added new type of embeddings: DocumentEmbedding.`

## Developing locally

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.

### Setup

Flair requires python-3.9 or higher. To make sure your code also runs on the oldest supported
python version, it is recommended to use python-3.9.x for flair development.

Create a python environment of your preference and run:
```bash
pip install -r requirements-dev.txt
pip install -e .
```

### Tests

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

### Code Formatting

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black) and for standardizing imports we use [ruff](https://github.com/charliermarsh/ruff).
If your code is not formatted properly, the tests will fail.

We recommend configuring your IDE to run these formatters for you, but you can also always run them manually via
`black . && ruff --fix .` in the flair root folder.
