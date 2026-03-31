project = "Slate Quantum"
author = "Matthew Ord"
version = "0.0.1"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "slate": ("https://matt-ord.github.io/slate/", None),
}


templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
