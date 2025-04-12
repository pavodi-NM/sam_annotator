# SAM Annotator Documentation Guide

This file explains how to build, view, and contribute to the SAM Annotator documentation.

## Viewing Documentation

### Online Documentation

The official documentation is available online at:
- https://pavodi-nm.github.io/sam_annotator/

### Documentation on PyPI

Basic documentation is available on the PyPI project page:
- https://pypi.org/project/sam-annotator/

## Building Documentation Locally

To build and view the documentation locally:

1. Install the required dependencies:

```bash
# Option 1: Install with pip
pip install -e ".[docs]"

# Option 2: Install dependencies directly
pip install mkdocs mkdocs-material mkdocstrings[python] mike
```

2. Serve the documentation locally:

```bash
mkdocs serve
```

This will start a local server at http://127.0.0.1:8000/ with live-reloading enabled.

3. Build static documentation:

```bash
mkdocs build
```

This creates a `site` directory with the HTML documentation.

## Documentation Structure

- `docs/index.md`: Main landing page
- `docs/shortcuts.md`: Keyboard shortcuts reference
- Additional documentation pages

## Contributing to Documentation

We welcome contributions to improve our documentation! Here's how to contribute:

1. Fork the repository
2. Make your changes to files in the `docs/` directory
3. Test your changes locally with `mkdocs serve`
4. Submit a pull request

### Documentation Style Guidelines

- Use clear, concise language
- Include examples where appropriate
- Use proper Markdown formatting
- Keep headings organized in a logical hierarchy
- Include screenshots for UI-related features

## Automatic Deployment

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch. The process is handled by the GitHub Actions workflow defined in `.github/workflows/docs.yml`. 