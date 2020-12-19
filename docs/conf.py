from pallets_sphinx_themes import get_version
from pallets_sphinx_themes import ProjectLink

# Project
project = "ir-metrics"
copyright = "2020 kqf"
author = "kqf"
release, version = get_version("ir-metrics", version_length=1)

# General
master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.log_cabinet",
    "pallets_sphinx_themes",
    "sphinx_issues",
    "sphinx_tabs.tabs",
    "numpydoc",
]
intersphinx_mapping = {"python": ("https://docs.python.org/3/", None)}
issues_github_path = "kqf/ir-metrics"

# HTML
html_theme = "flask"
html_theme_options = {"index_sidebar_logo": False}
html_context = {
    "project_links": [
        ProjectLink("Website", "https://ir-metrics.readthedocs.io"),
        ProjectLink("PyPI releases", "https://pypi.org/project/ir-metrics/"),
        ProjectLink("Source Code", "https://github.com/kqf/ir-metrics/"),
        ProjectLink("Issues", "https://github.com/kqf/ir-metrics/issues/"),
    ]
}
html_sidebars = {
    "index": ["project.html", "localtoc.html", "searchbox.html"],
    "**": ["localtoc.html", "relations.html", "searchbox.html"],
}
singlehtml_sidebars = {"index": ["project.html", "localtoc.html"]}
# html_static_path = ["_static"]
# html_favicon = "_static/ir-metrics-icon.png"
# html_logo = "_static/ir-metrics-logo-sidebar.png"
html_title = f"ir-metrics Documentation ({version})"
html_show_sourcelink = False

# LaTeX--
latex_documents = [(
    master_doc, f"irm-{version}.tex", html_title, author, "manual")]
