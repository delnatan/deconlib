# MkDocs + Material Setup Guide

A reusable guide for setting up modern Python documentation with MkDocs, Material theme, and mkdocstrings.

## Quick Start (5 Minutes)

### 1. Add Dependencies to `pyproject.toml`

```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.24",
]
```

### 2. Install

```bash
pip install -e ".[docs]"
```

### 3. Create `mkdocs.yml`

```yaml
site_name: your-project
site_description: Your project description
repo_url: https://github.com/username/your-project
repo_name: username/your-project

theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy

plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
            merge_init_into_class: true

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - admonition
  - pymdownx.details
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference: api.md
```

### 4. Create Minimal `docs/` Structure

```
docs/
├── index.md           # Home page
├── getting-started.md # Quick start guide
└── api.md             # API reference
```

### 5. Run Locally

```bash
mkdocs serve
# Open http://127.0.0.1:8000
```

### 6. Build Static Site

```bash
mkdocs build
# Output in site/
```

---

## Key Concepts

### Auto-Generated API Docs

Use `:::` directive to pull docs from docstrings:

```markdown
# API Reference

## MyClass

::: mypackage.module.MyClass

## my_function

::: mypackage.module.my_function
```

### Controlling What's Shown

```markdown
::: mypackage.MyClass
    options:
      show_source: false      # Hide source code
      members: true           # Show all members
      inherited_members: true # Include inherited
```

### Docstring Style

Supports Google, NumPy, or Sphinx style. Configure in `mkdocs.yml`:

```yaml
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google  # or numpy, sphinx
```

---

## Common Patterns

### Admonitions (Callouts)

```markdown
!!! note "Title"
    This is a note.

!!! warning
    This is a warning.

!!! tip "Pro Tip"
    This is a tip.
```

Renders as styled callout boxes.

### Code Blocks with Tabs

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "JavaScript"
    ```javascript
    console.log("Hello");
    ```
```

### Math Support (Optional)

Add to `mkdocs.yml`:

```yaml
markdown_extensions:
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - javascripts/mathjax.js
  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
```

Create `docs/javascripts/mathjax.js`:

```javascript
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
```

Then use: `\( E = mc^2 \)` for inline or `\[ E = mc^2 \]` for block.

---

## Deployment

### GitHub Pages (Recommended)

One command deployment:

```bash
mkdocs gh-deploy
```

This builds and pushes to `gh-pages` branch.

### GitHub Actions (Automatic)

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[docs]"

      - name: Deploy docs
        run: mkdocs gh-deploy --force
```

---

## Recommended Folder Structure

```
your-project/
├── pyproject.toml
├── mkdocs.yml
├── src/
│   └── your_package/
│       ├── __init__.py
│       ├── module.py
│       └── ...
├── docs/
│   ├── index.md                 # Home
│   ├── getting-started/
│   │   ├── installation.md
│   │   └── quickstart.md
│   ├── guide/
│   │   ├── feature1.md
│   │   └── feature2.md
│   ├── api/
│   │   ├── index.md             # API overview
│   │   ├── module1.md           # ::: directives
│   │   └── module2.md
│   └── javascripts/
│       └── mathjax.js           # If using math
└── tests/
```

---

## Tips

### 1. Write Good Docstrings First

mkdocstrings generates docs from your docstrings. Invest in good docstrings:

```python
def compute_psf(optics: Optics, shape: tuple[int, int]) -> np.ndarray:
    """Compute point spread function.

    Args:
        optics: Optical system parameters.
        shape: Output shape (ny, nx).

    Returns:
        PSF array normalized to sum to 1.

    Example:
        >>> psf = compute_psf(optics, (256, 256))
        >>> psf.shape
        (256, 256)
    """
```

### 2. Use `nav` for Organization

Explicit navigation in `mkdocs.yml` is clearer than auto-discovery:

```yaml
nav:
  - Home: index.md
  - User Guide:
      - Installation: guide/install.md
      - Usage: guide/usage.md
  - API: api/index.md
```

### 3. Preview Changes Live

```bash
mkdocs serve --dirtyreload
```

The `--dirtyreload` flag only rebuilds changed files (faster).

### 4. Check for Broken Links

```bash
mkdocs build --strict
```

Fails on warnings (broken links, missing refs).

---

## Customization

### Color Themes

Change `primary` and `accent` colors in `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: deep purple  # or: red, pink, indigo, blue, teal, green, etc.
    accent: amber
```

### Logo and Favicon

```yaml
theme:
  logo: assets/logo.png
  favicon: assets/favicon.png
```

### Social Links

```yaml
extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/username
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/username
```

---

## Troubleshooting

### "Module not found" errors

Make sure your package is installed:

```bash
pip install -e .
```

### Docstrings not rendering

Check docstring style matches config:

```yaml
options:
  docstring_style: google  # Must match your actual docstrings
```

### Build warnings

Run strict mode to see all issues:

```bash
mkdocs build --strict 2>&1 | head -50
```

---

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
