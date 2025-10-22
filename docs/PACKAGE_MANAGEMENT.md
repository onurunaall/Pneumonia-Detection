# Package Management & Development Guide

Complete guide for package management, Docker usage, CI/CD, and testing for the Pneumonia Detection project.

## Table of Contents
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [CI/CD](#cicd)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Package Structure](#package-structure)

---

## Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11
- pip 21.0+
- Git

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with all development dependencies
pip install -e ".[dev,test,api]"

# Or use make
make install-all
```

### Installing from Source

```bash
# Install from setup.py
python setup.py install

# Or build and install wheel
pip install build
python -m build
pip install dist/pneumonia_detection-1.0.0-py3-none-any.whl
```

---

## Requirements Files Structure

```
requirements/
├── requirements.txt          # Core production dependencies
├── requirements-dev.txt      # Development tools
├── requirements-test.txt     # Testing dependencies
└── requirements-api.txt      # API server dependencies
```

### Installing Specific Requirements

```bash
# Production only
pip install -r requirements/requirements.txt

# Development
pip install -r requirements/requirements-dev.txt

# Testing
pip install -r requirements/requirements-test.txt

# API
pip install -r requirements/requirements-api.txt
```

---

## Docker Setup

### Available Docker Images

The project provides multiple Docker images for different purposes:

1. **production** - Minimal production image for training
2. **api** - API server with FastAPI
3. **development** - Full development environment with Jupyter
4. **testing** - Test runner with coverage tools

### Building Docker Images

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build train
docker-compose build api

# Build production image only
make docker-build-prod

# Build API image only
make docker-build-api
```

### Running Docker Containers

#### Training
```bash
# Run training
docker-compose run --rm train

# Or with make
make train-docker
```

#### API Server
```bash
# Start API server
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

#### Development Environment
```bash
# Start development environment with Jupyter
docker-compose up -d dev

# Access Jupyter at http://localhost:8888
# Access TensorBoard at http://localhost:6006
```

#### Running Tests in Docker
```bash
# Run tests
docker-compose run --rm test

# Or with make
make docker-test
```

### Docker Compose Services

```yaml
# View all services
docker-compose ps

# Services:
# - train: Training service
# - api: API server (port 8000)
# - dev: Development environment (ports 8888, 6006)
# - test: Test runner
# - tensorboard: TensorBoard standalone (port 6006)
```

### Docker Commands Cheat Sheet

```bash
# Build all images
make docker-build

# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
make docker-logs

# Clean up
make docker-clean

# Run specific command in container
docker-compose run --rm train python -m scripts.train --epochs 10
```

### Docker Volumes

The Docker setup uses volumes for persistent data:

- `./data:/app/data` - Training data (read-only)
- `./models:/app/models` - Model checkpoints
- `./logs:/app/logs` - Training logs
- `./checkpoints:/app/checkpoints` - Model checkpoints

---

## CI/CD

### GitHub Actions

The project uses GitHub Actions for CI/CD with the following workflows:

#### Workflow Jobs

1. **lint** - Code quality checks
   - Black formatting
   - isort import sorting
   - flake8 linting
   - mypy type checking

2. **test** - Unit tests
   - Tests on Python 3.9, 3.10, 3.11
   - Coverage reports
   - Upload to Codecov

3. **security** - Security scanning
   - Safety (dependency vulnerabilities)
   - Bandit (code security issues)

4. **docker** - Docker image builds
   - Build production image
   - Build API image
   - Push to GitHub Container Registry

5. **integration** - Integration tests
   - Run Docker compose tests

6. **deploy** - Deployment (on release)
   - Deploy to production
   - Slack notifications

#### Triggering Workflows

```bash
# Workflows run automatically on:
# - Push to main/develop
# - Pull requests
# - Releases

# Manual trigger via GitHub UI:
# Actions → CI/CD Pipeline → Run workflow
```

#### GitHub Secrets Required

```
SLACK_WEBHOOK          # For deployment notifications
```

### GitLab CI

If using GitLab, the `.gitlab-ci.yml` provides:

#### Pipeline Stages

1. **lint** - Code quality
2. **test** - Unit tests (parallel across Python versions)
3. **security** - Security checks
4. **build** - Docker images
5. **deploy** - Staging/Production deployment

#### Running Locally

```bash
# Install GitLab Runner
# https://docs.gitlab.com/runner/install/

# Run pipeline locally
gitlab-runner exec docker test:unit
```

### CI/CD Best Practices

1. **All tests must pass before merge**
2. **Code coverage should not decrease**
3. **Docker images are tagged with commit SHA**
4. **Production deploys require manual approval**
5. **Failed builds block deployment**

---

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_dicom_utils.py      # DICOM utilities tests
├── test_preprocessing.py    # Preprocessing tests
├── test_model.py            # Model tests
├── test_dataset.py          # Dataset tests
└── test_predictor.py        # Inference tests
```

### Running Tests

#### Basic Test Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pneumonia_detection --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::TestImagePreprocessor::test_preprocess_grayscale_to_rgb

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_preprocess"
```

#### Using Make

```bash
# Run all tests
make test

# Run fast tests only (skip slow tests)
make test-fast

# Run with coverage report
make test-coverage

# Run failed tests from last run
make test-failed
```

#### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration
```

### Test Fixtures

Common fixtures available in `conftest.py`:

- `temp_dir` - Temporary directory
- `sample_image` - Sample grayscale image
- `sample_rgb_image` - Sample RGB image
- `sample_dicom` - Valid DICOM file
- `invalid_dicom` - Invalid DICOM file
- `sample_labels_df` - Sample labels DataFrame
- `mock_keras_model` - Mocked Keras model

### Writing Tests

```python
import pytest

def test_example(sample_image):
    """Test with sample image fixture"""
    assert sample_image.shape == (512, 512)

@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow"""
    pass

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
])
def test_parametrized(input, expected):
    """Parametrized test"""
    assert input * 2 == expected
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=pneumonia_detection --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Testing in Docker

```bash
# Run tests in Docker
docker-compose run --rm test

# Run specific tests
docker-compose run --rm test pytest tests/test_preprocessing.py
```

---

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Clone and install
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install development dependencies
make install-dev

# 4. Install pre-commit hooks
make pre-commit-install

# 5. Run tests to verify setup
make test
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality.

#### Installing Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

#### What Pre-commit Checks

- Code formatting (black, isort)
- Linting (flake8, pylint)
- Type checking (mypy)
- Security (bandit)
- File checks (trailing whitespace, large files, etc.)
- Notebook formatting

#### Bypassing Pre-commit

```bash
# Skip pre-commit hooks (not recommended)
git commit --no-verify -m "message"
```

### Code Formatting

```bash
# Format all code
make format

# Check formatting without changes
make format-check

# Format with black
black .

# Sort imports with isort
isort .
```

### Linting

```bash
# Run all linters
make lint

# Run specific linter
flake8 .
pylint pneumonia_detection
mypy .
```

### Development Commands

```bash
# Start Jupyter notebook
make jupyter

# Start TensorBoard
make tensorboard

# Run API in development mode
make api-run

# Profile code
make profile

# Security check
make security-check
```

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes
# ... code changes ...

# 3. Run tests
make test

# 4. Format and lint
make format
make lint

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add new feature"

# 6. Push
git push origin feature/new-feature

# 7. Create pull request
# ... via GitHub/GitLab UI ...
```

### Before Pushing

```bash
# Run all checks
make pre-push

# This runs:
# - Code formatting
# - Linting
# - Tests
```

---

## Package Structure

### Source Code Structure

```
pneumonia-detection/
│
├── pneumonia_detection/          # Main package
│   ├── __init__.py
│   ├── config/                   # Configuration
│   ├── data/                     # Data handling
│   ├── models/                   # Model architectures
│   ├── training/                 # Training utilities
│   ├── evaluation/               # Evaluation tools
│   ├── inference/                # Inference pipeline
│   └── utils/                    # Utilities
│
├── scripts/                      # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/                        # Test suite
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── requirements/                 # Requirement files
│
├── setup.py                      # Package setup
├── setup.cfg                     # Setup configuration
├── pyproject.toml               # Build system config
├── Makefile                      # Development commands
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker Compose config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .github/                      # GitHub Actions
│   └── workflows/
│       └── ci-cd.yml
└── .gitlab-ci.yml               # GitLab CI
```

### Package Configuration Files

- **setup.py** - Package installation and metadata
- **setup.cfg** - Tool configurations (flake8, pytest, etc.)
- **pyproject.toml** - Modern Python packaging (PEP 518)
- **requirements/** - Dependency management
- **Makefile** - Development task automation
- **.pre-commit-config.yaml** - Pre-commit hook configuration

### Entry Points

The package provides command-line scripts:

```bash
# After installation, these commands are available:
pneumonia-train          # Train model
pneumonia-predict        # Run inference
pneumonia-evaluate       # Evaluate model
```

---

## Common Tasks

### Training a Model

```bash
# Local
python -m scripts.train --epochs 50 --batch-size 32

# With make
make train

# In Docker
make train-docker
```

### Running Inference

```bash
# Local
python -m scripts.predict --model models/best_model.h5 --input data/test.dcm

# With make
make predict INPUT=data/test.dcm

# In Docker
docker-compose run --rm api
```

### Evaluating a Model

```bash
# Local
python -m scripts.evaluate

# With make
make evaluate
```

### Starting API Server

```bash
# Local
uvicorn api.main:app --reload

# With make
make api-run

# In Docker
make api-docker
```

---

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .
```

#### Docker Permission Issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Test Failures

```bash
# Update dependencies
pip install --upgrade -r requirements/requirements-test.txt

# Clear pytest cache
rm -rf .pytest_cache
```

#### Pre-commit Issues

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean
```

---

## Additional Resources

- **GitHub Actions Docs**: https://docs.github.com/actions
- **GitLab CI Docs**: https://docs.gitlab.com/ee/ci/
- **Docker Docs**: https://docs.docker.com
- **pytest Docs**: https://docs.pytest.org
- **pre-commit Docs**: https://pre-commit.com

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/pneumonia-detection/issues
- Email: your.email@example.com

---

## License

MIT License - see LICENSE file for details
