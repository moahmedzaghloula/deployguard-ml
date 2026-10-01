PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
BLACK := .venv/bin/black
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest
PYTHONPATH := src

.PHONY: help venv install-dev format format-check lint test check data validate-data eda data-pipeline train evaluate model-check

help:
	@printf "Available targets:\n"
	@printf "  venv          Create the local virtual environment\n"
	@printf "  install-dev   Install runtime and development dependencies\n"
	@printf "  format        Format Python source and tests\n"
	@printf "  format-check  Verify formatting without modifying files\n"
	@printf "  lint          Run Ruff lint checks\n"
	@printf "  test          Run Pytest\n"
	@printf "  check         Run all non-destructive quality checks\n"
	@printf "  data          Generate the deterministic deployment dataset\n"
	@printf "  validate-data Validate the generated dataset contract\n"
	@printf "  eda           Run EDA and save figures\n"
	@printf "  data-pipeline Generate, validate, and analyze the dataset\n"
	@printf "  train         Train, select, and save the model\n"
	@printf "  evaluate      Show the saved final evaluation\n"
	@printf "  model-check   Validate model artifacts and lineage\n"

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

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.generate_data

validate-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.validate_data

eda:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.eda

data-pipeline: data validate-data eda

train: validate-data
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.train train

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.train evaluate

model-check: validate-data
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m deployguard.train check

