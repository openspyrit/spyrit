# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#%%
import os
import sys
from sphinx_gallery.sorting import ExampleTitleSortKey
# paths relative to this file
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
#%%
project = "spyrit"
copyright = "2021, Antonio Tomas Lorente Mur - Nicolas Ducros - Sebastien Crombez - Thomas Baudier"
author = (
    "Antonio Tomas Lorente Mur - Nicolas Ducros - Sebastien Crombez - Thomas Baudier"
)

# The full version, including alpha/beta/rc tags
release = "2.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    'sphinx.ext.autodoc',
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.coverage"
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_member_order = "bysource"
autosummary_generate = True
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../../tutorial",
    ],
    # path where to save gallery generated examples
    "gallery_dirs": ["gallery"],
    "filename_pattern": "/tuto_",
    "ignore_pattern": "/_",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": ExampleTitleSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.
    "doc_module": "spyrit",
    # Insert links to documentation of objects in the examples
    "reference_url": {"spyrit": None},
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# By default, this is set to include the _static path.
html_static_path = []

# The master toctree document.
master_doc = "index"

html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}

# http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports
# autodoc_mock_imports = "numpy matplotlib mpl_toolkits scipy torch torchvision Pillow opencv-python imutils PyWavelets pywt wget imageio".split()


# Skip the following members when documenting (tutorials)
# https://stackoverflow.com/questions/3757500/connect-sphinx-autodoc-skip-member-to-my-function
# def autodoc_skip_member(app, what, name, obj, skip, options):
#     exclude_prefix = "tuto_"
#     if name.startswith(exclude_prefix):
#         return True
#     return None

# def setup(app):
#     app.connect("autodoc-skip-member", autodoc_skip_member)
