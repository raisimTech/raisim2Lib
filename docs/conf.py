extensions = [
    "breathe",
    'sphinx.ext.autosectionlabel',
    'sphinx_tabs.tabs'
]

html_theme = "sphinx_rtd_theme"

# Avoid accidentally treating the local venv (and other artifacts) as documentation sources.
exclude_patterns = [
    "_build",
    ".venv",
    ".venv/**",
]

# General information about the project.
project = 'raisim'
copyright = '2022, RaiSim Tech Inc.'
author = 'Yeonjoo Chung and Jemin Hwangbo'
version = '2.0.0'
release = '2.0.0'

# Output file base name for HTML help builder.
htmlhelp_basename = 'raisim_doc'
html_show_sourcelink = False

# Breathe Configuration
breathe_default_project = "raisim"
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 4
html_show_sphinx = False
html_logo = "image/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
}
suppress_warnings = [
    "cpp.duplicate_declaration",
    "c.duplicate_declaration",
]
