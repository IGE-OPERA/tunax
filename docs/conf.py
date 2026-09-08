# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.append(os.path.join('..', 'tunax'))
sys.path.append(os.path.join('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tunax'
author = 'Gabriel Mouttapa'
copyright = '2024, Gabriel Mouttapa, Julien Le Sommer'
version = '0.2'
release = '0.2.12'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    "sphinx.ext.duration",
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    "myst_parser",
    'nbsphinx'
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# autodoc_mock_imports = ["jax", "xarray", "optax", "netcdf4", "equinox", "jaxtyping"]
autodoc_default_options = {
#     'undoc-members': False,
#     'private-members': False,
#     'special-members': False,
#     'inherited-members': False,
#     'show-inheritance': True,
    'exclude-members': '__init__',
    'member-order': 'bysource'
}
# -- Napoleon configuration --------------------------------------------------
autodoc_class_signature = "separated"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'optax': ('https://optax.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}

# -- Layout configuration ----------------------------------------------------
add_module_names = False

# -- Options for markdown files ----------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
autosummary_generate = True
