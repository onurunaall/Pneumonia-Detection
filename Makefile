.PHONY: help install install-dev install-all clean clean-pyc clean-build clean-test lint format test test-cov test-fast docker-build docker-run docs serve-docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
DOCKER_IMAGE := pneumonia-detection
DOCKER_TAG := latest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies only
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all: ## Install all dependencies (dev + api + viz + mlops)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[all]"
	pre-commit install

# Cleaning targets
clean: clean-pyc clean-build clean-test ## Remove all build, test, coverage and Python artifacts

clean-pyc: ## Remove Python file artifacts
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	find . -name '*~' -delete

clean-build: ## Remove build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -type d -exec rm -rf {} +
	find . -name '*.egg' -type f -delete

clean-test: ## Remove test and coverage artifacts
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .mypy_cache/

# Code quality targets
lint: ## Check code quality with flake8 and mypy
	$(FLAKE8) config data models training evaluation inference utils scripts
	$(MYPY) config data models training evaluation inference utils

format: ## Format code with black and isort
	$(BLACK) .
	$(ISORT) .

format-check: ## Check if code is formatted correctly
	$(BLACK) --check .
	$(ISORT) --check-only .

# Testing targets
test: ## Run tests with pytest
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-fast: ## Run tests in parallel (requires pytest-xdist)
	$(PYTEST) tests/ -n auto -v

test-unit: ## Run only unit tests
	$(PYTEST) tests/ -v -m "unit"

test-integration: ## Run only integration tests
	$(PYTEST) tests/ -v -m "integration"

test-slow: ## Run all tests including slow ones
	$(PYTEST) tests/ -v -m "slow"

# Training targets
train: ## Train the model with default configuration
	$(PYTHON) scripts/train.py

train-debug: ## Train with debugging enabled
	$(PYTHON) scripts/train.py --debug

# Prediction targets
predict: ## Run prediction (requires MODEL and INPUT arguments)
	$(PYTHON) scripts/predict.py $(INPUT) --model $(MODEL)

# Docker targets
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t $(DOCKER_IMAGE):dev .

docker-run: ## Run Docker container
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-gpu: ## Run Docker container with GPU support
	docker run -it --rm --gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/logs:/app/logs \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-shell: ## Open shell in Docker container
	docker run -it --rm \
		-v $(PWD):/app \
		$(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# Documentation targets
docs: ## Generate documentation with Sphinx
	cd docs && make html

serve-docs: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Jupyter targets
notebook: ## Start Jupyter notebook server
	jupyter notebook notebooks/

lab: ## Start JupyterLab server
	jupyter lab notebooks/

# Pre-commit targets
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Package building targets
build: clean ## Build source and wheel distributions
	$(PYTHON) setup.py sdist bdist_wheel
	ls -lh dist/

upload-test: build ## Upload to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	$(PYTHON) -m twine upload dist/*

# Check targets
check: format-check lint test ## Run all checks (format, lint, test)

# Version management
version: ## Show current version
	@$(PYTHON) -c "from setup import setup; print(setup.version)"

bump-patch: ## Bump patch version (0.0.X)
	@echo "Bumping patch version..."
	@bump2version patch

bump-minor: ## Bump minor version (0.X.0)
	@echo "Bumping minor version..."
	@bump2version minor

bump-major: ## Bump major version (X.0.0)
	@echo "Bumping major version..."
	@bump2version major
