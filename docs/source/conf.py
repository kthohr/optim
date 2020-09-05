#!/usr/bin/env python3

import os
import subprocess

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    subprocess.call('cd ..; doxygen', shell=True)

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'navigation_depth': 4
}

def setup(app):
    app.add_stylesheet("main_stylesheet.css")

# extensions = ['breathe','sphinx.ext.mathjax']
extensions = ['breathe','sphinxcontrib.katex','sphinxcontrib.contentui']
breathe_projects = { 'optimlib': '../xml' }
templates_path = ['_templates']
html_static_path = ['_static']
source_suffix = '.rst'
master_doc = 'index'
project = 'OptimLib'
copyright = '2016-2020 Keith O\'Hara'
author = 'Keith O\'Hara'

exclude_patterns = []
highlight_language = 'c++'
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'statsdoc'
