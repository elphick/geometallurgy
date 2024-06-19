# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pathlib import Path
import os
import sys

import numpy as np
import pyvista
import plotly
from plotly.io._sg_scraper import plotly_sg_scraper
from sphinx_gallery.sorting import FileNameSortKey

from elphick import geomet

os.environ["PYVISTA_OFF_SCREEN"] = "True"
plotly.io.renderers.default = 'sphinx_gallery_png'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'geometallurgy'
copyright = '2024, Greg Elphick'
author = 'Greg Elphick'
version = geomet.__version__


path = os.path.abspath("../..")
sys.path.insert(0, path)

# -- pyvista configuration ---------------------------------------------------

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
pyvista.BUILDING_GALLERY = True  # necessary when building the sphinx gallery
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = np.array([1024, 768]) * 2

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary',  # to document the api
              'sphinx.ext.viewcode',  # to add view code links
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',  # for parsing numpy/google docstrings
              'sphinx_gallery.gen_gallery',  # to generate a gallery of examples
              'sphinx_autodoc_typehints',
              'myst_parser',  # for parsing md files
              'sphinx.ext.todo'
              ]

todo_include_todos = True
autosummary_generate = True

examples_dirs: list[str] = ['../../examples']
gallery_dirs: list[str] = [str(Path('auto_examples') / Path(d).stem) for d in examples_dirs]

sphinx_gallery_conf = {
    'filename_pattern': r'\.py',
    'ignore_pattern': r'(__init__)|(debug.*)|(pv.*)|(02_flowsheet_from_dataframe)\.py',
    'examples_dirs': examples_dirs,
    'gallery_dirs': gallery_dirs,
    'nested_sections': False,
    'download_all_examples': False,
    'within_subsection_order': 'FileNameSortKey',
    "image_scrapers": (pyvista.Scraper(), "matplotlib", plotly_sg_scraper),
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# to widen the page...
html_css_files = ['custom.css']
