# DocIE documentation (Sphinx, Furo theme).
#
# The Sphinx root is docs/ itself rather than a separate documentation/ tree:
# this repository already kept its documentation as Markdown here, and pointing
# Sphinx at it publishes those files unchanged instead of forking them into a
# parallel set of .rst that would drift. MyST parses the Markdown, so a
# contributor edits the same file whether they are reading it on GitHub or on
# the published site.

project = "DocIE"
author = "ADBI"
copyright = "2026, ADBI"
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

# Prefix section labels with their document, so two pages may both have an
# "Overview" heading without colliding -- which would otherwise be a warning,
# and the build treats warnings as errors.
autosectionlabel_prefix_document = True

# Anchors for h1-h3, so one page can link to a specific section of another.
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

exclude_patterns = [
    "_build",
    # LaTeX sources for partner-facing contracts. They are deliverables built
    # to PDF, not pages of this site, and their README would otherwise be an
    # orphan document -- a warning, and so a failed build.
    "integration/**",
    "README.md",
]

html_theme = "furo"
html_title = "DocIE"
html_static_path = ["_static"]
html_css_files = ["docie.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#4F46E5",
        "color-brand-content": "#4F46E5",
        "color-admonition-background": "#EEF2FF",
        "font-stack": "system-ui, -apple-system, 'Segoe UI', sans-serif",
        "font-stack--monospace": "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#A5B4FC",
        "color-brand-content": "#A5B4FC",
        "color-admonition-background": "#1E1B4B",
    },
}

mermaid_version = "11.4.1"
mermaid_init_config = {"startOnLoad": False, "securityLevel": "strict"}
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_height = "auto"

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
