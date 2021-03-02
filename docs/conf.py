# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'ivaps'
copyright = '2020, Richard Liu'
author = 'factoryofthesun'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinxcontrib.bibtex'
]

inheritance_graph_attrs = dict(rankdir="TB", size='"7.0, 10.0"',
                               fontsize=12, ratio='auto',
                               bgcolor='"#ffffff"', center='true', style='solid')
inheritance_node_attrs = dict(shape='ellipse', fontsize=12,
                              fontname="monspace", height=0.75)
napoleon_use_param = False
autosummary_generate = True
autodoc_default_options = {'members': None,
                           'show-inheritance': None,
                           'inherited-members': None,
                           'member-order': 'groupwise'}

mathjax_config = {
    'TeX': {
        'Macros': {
            'vec': [r'{\bf #1}', 1],
            'ldot': [r'\left\langle #1, #2 \right\rangle', 2],
            'E': r'\mathbb{E}',
            'T': r'\mathcal{T}',
            'argmin': r'\mathrm{argmin}',
            'argmax': r'\mathrm{argmax}',
            'Var': r'\mathrm{Var}'
        }
    }
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
