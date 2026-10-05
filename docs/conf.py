import re
from pathlib import Path

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


def read_raisim_version():
    root_cmake = Path(__file__).resolve().parents[1] / "CMakeLists.txt"
    contents = root_cmake.read_text(encoding="utf-8")
    for pattern in (
        r"project\(\s*raisim\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
        r"set\(\s*RAISIM_VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
    ):
        match = re.search(pattern, contents)
        if match:
            return match.group(1)
    raise RuntimeError(f"Could not parse RAISIM_VERSION from {root_cmake}")


# Keep Sphinx package metadata tied to the released binary selected by the
# root CMake project. An upcoming changelog entry may intentionally be one
# version ahead until the package version is bumped as part of the release.
version = read_raisim_version()
release = version
rst_epilog = f"""
.. |raisim_version| replace:: {version}
.. |raisim_version_title| replace:: v{version}
"""

# Output file base name for HTML help builder.
htmlhelp_basename = 'raisim_doc'
html_show_sourcelink = False

# Breathe Configuration
breathe_default_project = "raisim"
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 4
html_show_sphinx = False
html_logo = "../rsc/docs/image/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo_only": True,
}
suppress_warnings = [
    "cpp.duplicate_declaration",
    "c.duplicate_declaration",
]
