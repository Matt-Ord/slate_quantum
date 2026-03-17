# SLATE Quantum

## Building Documentation

Documentation is built with Sphinx and follows the same steps used in CI
([.github/workflows/deploy.yml](.github/workflows/deploy.yml)).

### Build locally

From the repository root:

```bash
# Automatically generate API docs from docstrings
uv run sphinx-apidoc -o docs/source/ slate_quantum/
# Build the docs
cd docs
make html
```

The generated site is written to:

- `docs/build/html/`

Open `docs/build/html/index.html` in a browser to preview the docs locally.

### CI and deployment

- On pushes to `main`, GitHub Actions builds the docs and deploys them to GitHub Pages via
  [.github/workflows/deploy.yml](.github/workflows/deploy.yml).
- Pull requests to `main` run lint/type/test checks via
  [.github/workflows/test.yml](.github/workflows/test.yml).
