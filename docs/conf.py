# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Thucia"
copyright = "2025, kraemer-lab"
author = "kraemer-lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinx_book_theme",
    "sphinxcontrib.mermaid",
]

templates_path = [
    "_templates",
]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "venv",
    "README.md",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "images/logo.png"
