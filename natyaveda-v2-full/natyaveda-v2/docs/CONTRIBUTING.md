# Contributing to NatyaVeda Analyzer

Thank you for your interest in contributing! This document covers how to get started.

## Ways to Contribute

- **Dataset:** Annotate new dance videos; add underrepresented dance styles
- **Mudra taxonomy:** Improve or extend the hasta classification labels
- **Model improvements:** Experiment with new architectures or training strategies  
- **Bug fixes:** Fix issues reported in GitHub Issues
- **Documentation:** Improve the guides, docstrings, or notebooks

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/natyaveda-analyzer.git
cd natyaveda-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- **Formatter:** `black` (line length 100)
- **Imports:** `isort`
- **Linter:** `flake8`

Run all checks:
```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/ --max-line-length=100
```

## Running Tests Before a PR

```bash
pytest tests/ -v --tb=short
```

All 40+ tests must pass. New features should include corresponding tests.

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Code formatted (`black`, `isort`)
- [ ] Docstrings added for new public functions/classes
- [ ] GUIDE.md updated if new commands or workflows are added
- [ ] PR description explains what changed and why

## Reporting Issues

Please include:
- OS and Python version
- GPU model and CUDA version  
- Full error traceback
- Minimal reproduction steps
