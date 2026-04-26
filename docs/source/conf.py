# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import tomllib
# Add the src directory to the path so we can import vampire
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# Read version from pyproject.toml
def get_version_from_pyproject():
    """Read version from pyproject.toml"""
    pyproject_path = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
    try:
        with open(pyproject_path, 'rb') as f:
            pyproject = tomllib.load(f)
            return pyproject['project']['version']
    except (FileNotFoundError, KeyError):
        # Fallback to default version if file not found or parsing fails
        return '0.3.0'

# -- Project information -----------------------------------------------------

project = 'VAMPIRE'
copyright = '2026, VAMPIRE Contributors'
author = 'VAMPIRE development team'
# The full version, including alpha/beta/rc tags (automatically read from pyproject.toml)
release = get_version_from_pyproject()
# The short X.Y version (extract major.minor from release)
version = '.'.join(release.split('.')[:2])
repository_url = "https://github.com/Zikun-Yang/VAMPIRE"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "myst_nb",
    "sphinx_design",
]

# -- Options for myst-nb (Jupyter Notebook support) --------------------------

# Execution mode: "auto" = execute if no outputs present, "force" = always execute,
# "off" = never execute, "cache" = execute and cache outputs.
nb_execution_mode = "auto"

# Exclude patterns from execution (e.g., long-running notebooks)
# nb_execution_excludepatterns = ["tutorials/analysis/*.ipynb"]

# Timeout for notebook execution (seconds)
nb_execution_timeout = 300

# Render image options for notebook outputs
nb_render_image_options = {"width": "100%"}

# Allow plotly interactive outputs in HTML
nb_mime_priority_overrides = [
    ("html", "application/vnd.plotly.v1+json", 10),
    ("html", "image/png", 5),
]

# Generate the API documentation when building

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {
    # Don’t show members in addition to the autosummary table added by `_templates/class.rst`
    "members": False,
    # show “Bases: SomeClass” at the top of class docs
    "show-inheritance": True,
}

# -- Options for Napoleon ----------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]

# -- Options for HTML output -------------------------------------------------

html_theme = 'scanpydoc'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sphinx = False
html_title = "VAMPIRE"
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,

    "logo": {
        "image_light": "_static/img/VAMPIRE_logo_with_name_bright.png",
        "image_dark": "_static/img/VAMPIRE_logo_with_name_dark.png",
    },
}

# -- Options for plot -------------------------------------------------------

"""plot_include_source = True
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_html_show_source_link = False
plot_working_directory = ROOT  # Project root"""