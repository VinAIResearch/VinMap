# Copyright (c) OpenMMLab. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import subprocess
import sys

import pytorch_sphinx_theme


sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "MMOCR"
copyright = "2020-2030, OpenMMLab"
author = "OpenMMLab"

# The full version, including alpha/beta/rc tags
version_file = "../../mmocr/version.py"
with open(version_file, "r") as f:
    exec(compile(f.read(), version_file, "exec"))
__version__ = locals()["__version__"]
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "myst_parser",
]

autodoc_mock_imports = ["mmcv._ext"]

# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    "logo_url": "https://mmocr.readthedocs.io/en/latest/",
    "menu": [
        {
            "name": "Tutorial",
            "url": "https://colab.research.google.com/github/" "open-mmlab/mmocr/blob/main/demo/MMOCR_Tutorial.ipynb",
        },
        {"name": "GitHub", "url": "https://github.com/open-mmlab/mmocr"},
        {
            "name": "Upstream",
            "children": [
                {
                    "name": "MMCV",
                    "url": "https://github.com/open-mmlab/mmcv",
                    "description": "Foundational library for computer vision",
                },
                {
                    "name": "MMDetection",
                    "url": "https://github.com/open-mmlab/mmdetection",
                    "description": "Object detection toolbox and benchmark",
                },
            ],
        },
        {
            "name": "Version",
            "children": [
                {
                    "name": "MMOCR 0.x",
                    "url": "https://mmocr.readthedocs.io/en/latest/",
                    "description": "docs at main branch",
                },
                {
                    "name": "MMOCR 1.x",
                    "url": "https://mmocr.readthedocs.io/en/dev-1.x/",
                    "description": "docs at 1.x branch",
                },
            ],
            "active": True,
        },
    ],
    # Specify the language of shared menu
    "menu_lang": "en",
    "header_note": {
        "content": "You are reading the documentation for MMOCR 0.x, which "
        "will soon be deprecated by the end of 2022. We recommend you upgrade "
        "to MMOCR 1.0 to enjoy fruitful new features and better performance "
        " brought by OpenMMLab 2.0. Check out the "
        '<a href="https://mmocr.readthedocs.io/en/dev-1.x/migration/overview.html">maintenance plan</a>, '  # noqa
        '<a href="https://github.com/open-mmlab/mmocr/releases">changelog</a>, '  # noqa
        '<a href="https://github.com/open-mmlab/mmocr/tree/1.x">code</a> '  # noqa
        'and <a href="https://mmocr.readthedocs.io/en/dev-1.x/">documentation</a> of MMOCR 1.0 for more details.',  # noqa
    },
}

language = "en"

master_doc = "index"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/readthedocs.css"]

myst_heading_anchors = 3


def builder_inited_handler(app):
    subprocess.run(["./merge_docs.sh"])
    subprocess.run(["./stats.py"])


def setup(app):
    app.connect("builder-inited", builder_inited_handler)
