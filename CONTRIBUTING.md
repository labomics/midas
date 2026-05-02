# Contributing to MIDAS

Thanks for your interest in improving MIDAS. Bug reports, documentation fixes, and pull requests are all welcome.

## Reporting bugs and requesting features

Please open a [GitHub issue](https://github.com/labomics/midas/issues). For bug reports, the more of the following you can provide, the faster we can help:

- A minimal script that reproduces the problem.
- The full traceback (not just the last line).
- `python -c "import scmidas; print(scmidas.__version__)"`, your `torch.__version__`, and the OS/Python version.
- For DDP-related issues, the value of `STRATEGY` and `GPUS` from your demo cell.

## Development setup

```bash
git clone https://github.com/labomics/midas.git
cd midas
conda create -n scmidas python=3.12
conda activate scmidas
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, `mypy`, and `build`.

## Running the test suite

```bash
pytest tests/
```

CI runs the same suite on Python 3.10, 3.11, and 3.12 — please make sure `pytest` passes locally before opening a pull request.

## Pull request workflow

1. Branch from `main`.
2. Keep each PR focused on a single logical change. Multiple unrelated fixes in one PR slow down review and complicate revert.
3. Make sure `pytest tests/` is green.
4. For non-trivial changes (new public API, behaviour change, refactor that touches >1 module), open an issue first to discuss the design — it's much cheaper to align on direction before code than after.
5. Update `docs/source/release.md` under the next "Unreleased" / version heading.
6. Open the pull request against `main`. CI will run automatically; once it's green, a maintainer will review.

## Coding style

- Follow the existing patterns in the file you're editing rather than introducing a new style.
- `ruff` is configured (line-length 100, target Python 3.10) and runs in CI as a non-blocking lint. Code that's already clean on `ruff check .` is appreciated.
- New public APIs should have docstrings (NumPy or Google style — both are configured in Sphinx).
- Avoid module-level `logging.basicConfig(...)` in library code.

## Questions

For questions that aren't bug reports, prefer [GitHub Discussions](https://github.com/labomics/midas/discussions) (or open an issue if Discussions isn't enabled).
