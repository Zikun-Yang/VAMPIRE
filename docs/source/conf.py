# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import tomllib
# Add the src directory to the path so we can import vampire
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VAMPIRE'
copyright = '2026, VAMPIRE Contributors'
author = 'Zikun Yang, Shilong Zhang, Yafei Mao'
# The full version, including alpha/beta/rc tags (automatically read from pyproject.toml)
release = get_version_from_pyproject()
# The short X.Y version (extract major.minor from release)
version = '.'.join(release.split('.')[:2])
repository_url = "https://github.com/Zikun-Yang/VAMPIRE"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_design",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
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
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True

# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


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