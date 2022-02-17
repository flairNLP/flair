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

### setup

You can either use [Pipenv](https://pipenv.readthedocs.io/) for this:

```bash
pipenv install --dev && pipenv shell
```

or create a python environment of your preference and run
```bash
pip install -r requirements-dev.txt
```

### tests

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

### code formatting

To ensure a standardized code style we use the formatter [black](https://github.com/ambv/black) and for standardizing imports we use [isort](https://github.com/PyCQA/isort).
If your code is not formatted properly, the tests will fail.
simply execute
You can automatically format the code via `black --config pyproject.toml flair/ && isort flair/` in the flair root folder.

### pre-commit hook

If you want to automatically format your code on every commit, you can use [pre-commit](https://pre-commit.com/).
Just install it via `pip install pre-commit` and execute `pre-commit install` in the root folder.
This will add a hook to the repository, which reformats files on every commit.