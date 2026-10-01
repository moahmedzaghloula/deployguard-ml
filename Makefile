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


IMAGE ?= deployguard-ml:1.0.0
CONTAINER ?= deployguard-api
PORT ?= 8000
VCS_REF := $(shell git rev-parse --short=12 HEAD)
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

.PHONY: api docker-build docker-run docker-smoke docker-inspect

api: model-check
	PYTHONPATH=$(PYTHONPATH) .venv/bin/gunicorn --workers 2 --bind 127.0.0.1:8000 --access-logfile - --error-logfile - --worker-tmp-dir /tmp --timeout 30 --graceful-timeout 30 --keep-alive 5 --preload 'deployguard.app:create_app()'

docker-build: model-check
	docker build --pull --build-arg VCS_REF="$(VCS_REF)" --build-arg BUILD_DATE="$(BUILD_DATE)" --tag "$(IMAGE)" .

docker-run:
	docker image inspect "$(IMAGE)" >/dev/null
	docker run --detach --name "$(CONTAINER)" --publish "127.0.0.1:$(PORT):8000" --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m --cap-drop ALL --security-opt no-new-privileges=true --memory 512m --cpus 1.0 "$(IMAGE)"

docker-smoke:
	test "$$(docker inspect --format '{{.State.Health.Status}}' "$(CONTAINER)")" = "healthy"
	curl --fail --silent --show-error "http://127.0.0.1:$(PORT)/health" >/dev/null
	curl --fail --silent --show-error "http://127.0.0.1:$(PORT)/ready" >/dev/null
	curl --fail --silent --show-error --request POST --header 'Content-Type: application/json' --data '{"files_changed":32,"lines_added":850,"lines_deleted":180,"test_coverage_percent":71.5,"failed_tests":2,"previous_deployment_failures":1,"deployment_hour":22,"is_weekend":1,"team_experience_months":18}' "http://127.0.0.1:$(PORT)/predict" >/dev/null
	@printf "Docker smoke test: PASS\n"

docker-inspect:
	@docker image inspect "$(IMAGE)" --format 'Image={{.Id}} User={{.Config.User}}'
	@docker inspect "$(CONTAINER)" --format 'Health={{.State.Health.Status}} ReadOnly={{.HostConfig.ReadonlyRootfs}} CapDrop={{json .HostConfig.CapDrop}} SecurityOpt={{json .HostConfig.SecurityOpt}} Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}}'

HADOLINT_IMAGE ?= hadolint/hadolint:v2.15.1-debian

TRIVY_IMAGE ?= aquasec/trivy:0.74.0
TRIVY_CACHE_VOLUME ?= deployguard-trivy-cache

install: venv
	$(PIP) install --requirement requirements.lock.txt

coverage:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) --cov=deployguard --cov-report=term-missing --cov-report=xml:coverage.xml


security: docker-build
	@mkdir -p reports/ci
	docker run --rm -i $(HADOLINT_IMAGE) < Dockerfile
	docker run --rm -v "$(CURDIR):/work:ro" $(TRIVY_IMAGE) fs --scanners secret --no-progress --exit-code 1 --skip-dirs /work/.git --skip-dirs /work/.venv /work
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$(TRIVY_CACHE_VOLUME):/root/.cache/trivy" $(TRIVY_IMAGE) image --scanners vuln --no-progress --exit-code 0 --severity UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL $(IMAGE) > reports/ci/trivy-full.txt
	@echo "Full Trivy report: reports/ci/trivy-full.txt"
	@sed -n '1,40p' reports/ci/trivy-full.txt
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$(TRIVY_CACHE_VOLUME):/root/.cache/trivy" $(TRIVY_IMAGE) image --scanners vuln --no-progress --ignore-unfixed --exit-code 1 --severity HIGH,CRITICAL $(IMAGE)
	@echo "Container security checks: PASS"

.PHONY: verify

verify: format-check lint coverage data validate-data train model-check security
	@set -eu; \
	CONTAINER_NAME="$(CONTAINER)"; \
	docker rm -f "$$CONTAINER_NAME" >/dev/null 2>&1 || true; \
	cleanup() { \
		docker rm -f "$$CONTAINER_NAME" >/dev/null 2>&1 || true; \
	}; \
	trap cleanup EXIT INT TERM; \
	make docker-run; \
	status="starting"; \
	for attempt in $$(seq 1 30); do \
		status="$$(docker inspect --format '{{.State.Health.Status}}' "$$CONTAINER_NAME")"; \
		echo "Container health: $$status"; \
		if [ "$$status" = "healthy" ]; then \
			break; \
		fi; \
		sleep 1; \
	done; \
	test "$$status" = "healthy"; \
	make docker-smoke; \
	make docker-inspect; \
	echo "DeployGuard ML verification: PASS"
