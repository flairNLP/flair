# Contributing to Flair

We are happy to accept your contributions to make `flair` better and more awesome! To avoid unnecessary work on either 
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/zalandoresearch/flair/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the 
   commit guidelines below.

In case you just want to help out and don't know where to start, 
[issues with "help wanted" label](https://github.com/zalandoresearch/flair/labels/help%20wanted) are good for 
first-time contributors. 

## Git Commit Guidelines

If there is already a ticket, use this number at the start of your commit message. 
Use meaningful commit messages that described what you did.

**Example:** `GH-42: Added new type of embeddings: DocumentEmbedding.` 

## Developing locally

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.

### Development Setup

1) Install and select python-3.7, python-3.8 or python-3.9 (Using pyenv on linux)
2) Install the poetry dependency manager from https://python-poetry.org
   - osx / linux / bashonwindows: `curl -sSL https://install.python-poetry.org | python3 -`
   - windows powershell: `(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -`
3) Add poetry to your PATH
   - osx / linux add: $HOME/.local/bin
   - windows add: %APPDATA%\Python\Scripts
4) Install flair dependencies in a virtual environment and enter shell:
```bash
poetry install && poetry shell
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

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black) and for standardizing imports we use [isort](https://github.com/PyCQA/isort).
If your code is not formatted properly, the tests will fail.
You can automatically format the code via `black flair && isort flair` in the flair root folder.

### Pre-commit Hook

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.