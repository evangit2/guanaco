# Contributing to Guanaco

First off, thank you for considering contributing to Guanaco! It's people like you that make Guanaco such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Bug reports are hugely important. Before creating a bug report, please check the existing issues to avoid duplicates.

When filing a bug report, please include:

- **A clear, descriptive title**
- **Steps to reproduce** — the more specific, the better
- **Expected behavior** — what did you expect to happen?
- **Actual behavior** — what happened instead?
- **Environment details** — OS, Python version, Guanaco version
- **Logs** — any relevant log output or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome. Please include:

- **A clear, descriptive title**
- **Use case** — why is this enhancement useful?
- **Proposed solution** — how should it work?
- **Alternatives considered** — what other approaches have you thought of?

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-new-feature`)
3. Make your changes
4. Add tests for your changes if applicable
5. Ensure all tests pass (`pytest`)
6. Commit with a clear message (see [Commit Messages](#commit-messages))
7. Push to your fork (`git push origin feature/my-new-feature`)
8. Open a Pull Request against the `master` branch

PRs should:

- Address one concern at a time (keep them focused)
- Include tests for new functionality
- Update documentation for changed behavior
- Pass all existing tests

## Development Setup

```bash
# Clone the repository
git clone https://github.com/evangit2/guanaco.git
cd guanaco

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run the CLI
guanaco --help

# Run tests
pytest
```

### Running Locally

```bash
# Start the proxy server
guanaco serve

# Or use the short alias
oct serve
```

## Coding Standards

- **Python 3.10+** — use modern Python features (type hints, match statements, etc.)
- **Follow PEP 8** — use a linter/formatter (ruff, black, or flake8)
- **Type hints** —annotate function signatures where practical
- **Docstrings** — use docstrings for public modules, classes, and functions
- **Keep it async** — the codebase uses async/await; prefer async patterns for I/O
- **No secrets in code** — use environment variables or config files (never hardcode credentials)

## Commit Messages

- Use the present tense ("add feature" not "added feature")
- Use the imperative mood ("move cursor to..." not "moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and PRs when relevant

Thank you for contributing!