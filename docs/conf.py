# noqa: INP001

import importlib_metadata

# -- Project information -----------------------------------------------------
from sphinx_github_style import get_linkcode_resolve

version = "0.14.0"
release = "0.14.0"
project = "flair"
author = importlib_metadata.metadata(project)["Author"]
copyright = f"2023 {author}"

# The full version, including alpha/beta/rc tags
top_level = project.replace("-", "_")

linkcode_url = importlib_metadata.metadata(project)["Home-page"]
html_show_sourcelink = True

smv_current_version = ""  # will by overwritten by sphinx-multi-version to the name of the tag or branch.
html_context = {
    "display_github": True,
    "github_user": "flairNLP",
    "github_repo": "flair",
    "github_version": "",
    "conf_py_path": "/docs/",
}  # dummy value that sphinx-github-style won't crash when run in temp folder.

html_theme_options = {
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "github_url": linkcode_url,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/flair",
            "icon": "fas fa-box",
        },
    ],
}


def linkcode_resolve(*args):
    # use smv_current_version as the git url
    real_linkcode_url = linkcode_url + f"/blob/{smv_current_version}/" + "{filepath}#L{linestart}-L{linestop}"
    return get_linkcode_resolve(real_linkcode_url)(*args)


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",  # to render Google format docstrings
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_github_style",
    "sphinx_autodoc_typehints",
    "sphinx_multiversion",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Napoleon settings
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "members": True,
    "show-inheritance": True,
    "private-members": False,
    "inherited": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_sidebars = {
    "**": [
        "globaltoc.html",
        "searchbox.html",
        "versioning.html",
    ],
    "index": [],
}

smv_latest_version = importlib_metadata.version(project)

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r"^master$"

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = r"^origin$"

# Pattern for released versions
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"

# Format for versioned output directories inside the build directory
smv_outputdir_format = "{ref.name}"

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False

html_favicon = "_static/favicon.ico"
