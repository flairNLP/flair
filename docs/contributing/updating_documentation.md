# Updating documentation


## What is good documentation?

Good Documentation
* Always refers to the enduser. Do not document *why* something is the way it is, but rather *how* to use it.
* Doesn't lie and is always up-to-ate. Whenever code is updated, consider if the documentation needs to change accordingly to reflect reality.
* Provides useful links whenever usable. Do not reference another object without linking it.


## Tutorials

All tutorials are markdown files stored at [the tutorial folder](https://github.com/flairNLP/flair/tree/master/docs/tutorial).
When adding a new tutorial, you must add its name to the `index.rst` file in the respective folder.
We are using the [MyST parser](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html) which adds
some additional syntax over markdown.

A tutorial should always be easy to understand, and reference api documentation for future readings.

```{note}
  You can reference symbols by defining links
  e.g.: ``[`flair.set_seed`](#flair.set_seed)`` for a function
  e.g.: `[entity-linking](project:../tutorial/tutorial-basics/entity-linking.md)` for another tutorial
```

## Docstrings

For docstrings we follow the [Google docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) format.
We do not need to specify types or default values, as those will be extracted from the function signature.

Docstrings have usual a 1 liner giving a simple explanation of the object. Then there is a more detailed explanation followed **if required**.
Ensure that you always use cross-references instead of just mentioning another object,
e.g. ``:class:`flair.models.SequenceTagger` `` can be used to reference the SequenceTagger.


## Building the local docs

For building the docs,

* Ensure that you have everything committed. Local changes won't be used for building.
* Install the build dependencies via `pip install -r docs/requirements.txt`.
* In `docs/conf.py` temporarily add your local branch name to the `smv_branch_whitelist` pattern. 
  E.g. if your branch is called `doc-page` `smv_branch_whitelist` need to have the value `r"^master|doc-page$"`
* run `sphinx-multiversion docs doc_build/` to generate the docs.
* open `doc_build/<your branch name>/index.html` to view the docs.
