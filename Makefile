PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
BLACK := .venv/bin/black
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

.PHONY: help venv install-dev format format-check lint test check

help:
	@printf "Available targets:\n"
	@printf "  venv          Create the local virtual environment\n"
	@printf "  install-dev   Install runtime and development dependencies\n"
	@printf "  format        Format Python source and tests\n"
	@printf "  format-check  Verify formatting without modifying files\n"
	@printf "  lint          Run Ruff lint checks\n"
	@printf "  test          Run Pytest\n"
	@printf "  check         Run all non-destructive quality checks\n"

venv:
	test -d .venv || python3 -m venv .venv

install-dev: venv
	$(PIP) install -r requirements-dev.in

format:
	$(BLACK) src tests

format-check:
	$(BLACK) --check src tests

lint:
	$(RUFF) check src tests

test:
	$(PYTEST)

check: format-check lint test
