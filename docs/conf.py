# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# --- Metadatos del proyecto ---
project = 'Curso introducción a GTPs y su aplicación en la programación'
author = 'MCD. Patricia Muñoz'
release = '0.1'

# --- Extensiones ---
extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
]

# MyST
myst_enable_extensions = [
    "colon_fence",
    "dollarmath"
]

# Rutas y exclusiones
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    '**/.ipynb_checkpoints',
    'notebooks/*.ipynb',
    '**/*.html',
]

# --- Tema ---
html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/patymunoz/curso-gpt-programacion",
    "repository_branch": "main",   
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": False,
    "home_page_in_toc": True,
}

html_title = "Curso introducción a GTPs y su aplicación en la programación"
html_logo = "_static/cerebro.png" 
html_static_path = ['_static']

# Reconocer archivos .md como Markdown
source_suffix = {
    '.md': 'markdown',
}