# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import adamops

from importlib.metadata import version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AdamOps'
copyright = '2026, AdamOps Team'
author = 'AdamOps Team'
try:
    release = version('adamops')
except Exception:
    release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Extract documentation from Python docstrings
    'sphinx.ext.napoleon',      # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.intersphinx',   # Link to other projects' documentation
    'myst_parser',              # Allow writing Markdown in addition to reST
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Document both class docstrings and __init__ docstrings
autoclass_content = 'both'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
