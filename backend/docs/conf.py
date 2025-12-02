import os
import sys

DOCS_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, BACKEND_ROOT)  # interpy_bg, interpy_synth
sys.path.insert(0, os.path.join(BACKEND_ROOT, "fivedreg_tf"))  # fivedreg_tf

# -- Project information -----------------------------------------------------
project = 'InterPyApp'
copyright = '2025, Baron Gracias'
author = 'Baron Gracias'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",          # Automatically document modules/classes/functions
    "sphinx_autodoc_typehints",    # Show Python type hints in docs
    "sphinx.ext.napoleon",         # Google & NumPy style docstrings
    "sphinx.ext.viewcode",         # Link to source code
    "sphinx.ext.autosummary"       # Generate summary tables for classes/functions
]

# Automatically document members of classes and modules
autodoc_default_options = {
    "members": True,           # Show all members
    "undoc-members": True,     # Show members even if they have no docstring
    "inherited-members": True, # Include inherited methods
    "show-inheritance": True   # Show class inheritance
}

# Enable autosummary generation
autosummary_generate = True

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for linking to source code on GitHub ----------------------------
# Update with your repository info
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "barongracias",
    "github_repo": "InterPyApp",
    "github_version": "main",
    "conf_py_path": "/backend/docs/",  # Path to docs folder in repo
}
