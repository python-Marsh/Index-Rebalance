# Index-Rebalance
This repo builds a model that replicates FTSE 100 constituent via live data.

## Pre-commit Usage

This project uses **pre-commit** hooks to automatically format code, check imports, and enforce style guidelines before committing changes.

### Setup
Install the git hooks in your local repository:
```
poetry run pre-commit install
```
This sets up the hooks to run automatically on git commit.

### Running Pre-commit Manually
You can run all pre-commit hooks against all files to check and fix issues at any time:
```
poetry run pre-commit run --all-files
```

### Commit Without Running Pre-commit Hooks
If you want to bypass pre-commit hooks and commit your changes directly, you can use:
```
git commit --no-verify -m "Your commit message"
```
Note: Bypassing hooks is generally discouraged unless you have a specific reason to do so.

### What the Hooks Do
- **black**: Automatically formats Python code.

- **isort**: Sorts and organizes imports.

- **docformatter**: Formats docstrings to comply with style guidelines.

- **flake8**: Lints Python code for style violations, including line length, unused imports, and more.

### Troubleshooting
If pre-commit modifies files, review the changes, stage them, and commit again.

Ensure your code lines respect the max length (default 88 characters) to avoid flake8 errors.

Update hooks by running:
```
poetry run pre-commit autoupdate
```