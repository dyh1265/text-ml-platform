# How to Write Clean Python Projects Recruiters Love

A practical guide to structuring Python projects so they look professional, maintainable, and impressive to technical recruiters and hiring managers.

---

## 1. Project Structure

```
project-root/
├── src/
│   ├── __init__.py
│   ├── package_a/
│   │   ├── __init__.py
│   │   └── module.py
│   └── package_b/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── docs/
├── scripts/
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .pre-commit-config.yaml
├── .env.example
├── LICENSE
└── README.md
```

**Why recruiters care:** A clear layout shows you think about organization and maintainability. `src/` as the top-level package avoids import issues and supports editable installs.

---

## 2. Essential Files

### `pyproject.toml`

Single source of truth for metadata, dependencies, and tool config.

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "One-line description."
requires-python = ">=3.10"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Your Name" }]
keywords = ["relevant", "keywords"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "integration: marks tests requiring external services",
]

[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "W", "F", "I", "UP", "B", "SIM", "RUF"]
```

### `LICENSE`

Use MIT or Apache 2.0 for most open-source projects. No license = unclear terms for use and contribution.

### `.env.example`

Document all configuration variables with sensible defaults. Never commit real secrets.

```bash
# Database
DATABASE_URL=postgresql://localhost:5432/mydb

# API
API_KEY=your-key-here
API_TIMEOUT=30
```

---

## 3. Code Quality

### Type hints

Use modern PEP 604 style:

```python
# Prefer
def process(items: list[str] | None = None) -> dict[str, int]:
    ...

# Avoid
def process(items: Optional[List[str]] = None) -> Dict[str, int]:
    ...
```

### Structured logging, not `print()`

```python
# Prefer
log_event("job_started", job_id=id, count=len(items))

# Avoid
print(f"Starting job {id} with {len(items)} items")
```

Logs should be machine-parseable (e.g. JSON) for production and include timestamps and severity.

### Narrow exception handling

```python
# Prefer
except requests.RequestException as e:
    log_event("request_failed", error=str(e))

# Avoid
except Exception as e:
    pass
```

Catch specific exceptions. Use `# noqa: BLE001` only when a broad catch is intentional and documented.

### DRY (Don't Repeat Yourself)

Extract shared logic (e.g. Kafka producer creation, schema definitions) into utility modules. Duplicated code is a red flag.

---

## 4. Testing

- **Coverage:** Aim for ≥40% on core logic. Use `pytest-cov` and `--cov-report=term-missing`.
- **Unit tests:** Mock external services (HTTP, DB, Kafka). Tests should be fast and deterministic.
- **Integration tests:** Mark with `@pytest.mark.integration` and exclude from default runs.
- **`conftest.py`:** Add project root to `sys.path` so imports work; define shared fixtures.

```python
# conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

---

## 5. CI/CD

Use GitHub Actions (or similar) to run lint and tests on every push/PR.

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check src/ tests/
      - run: ruff format --check src/ tests/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt pytest-cov
      - run: pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 6. Pre-commit Hooks

Catch issues before commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-merge-conflict
```

Run: `pre-commit install`

---

## 7. Makefile

Provide simple, predictable commands:

```makefile
.PHONY: lint test format clean

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
```

---

## 8. README

Include:

- **Badges:** CI status, Python version, license, code style
- **One-paragraph description** of what the project does and why it matters
- **Architecture diagram** (e.g. Mermaid) for non-trivial systems
- **Quick start** (3–5 commands to run locally)
- **Configuration table** (env vars, defaults, descriptions)
- **Tests:** How to run them
- **License:** Link to LICENSE file

---

## 9. Configuration

Centralize config in one module (e.g. `src/config.py`) and read from environment variables. Never hardcode secrets or environment-specific values.

```python
# config.py
import os

API_URL: str = os.getenv("API_URL", "https://api.example.com")
DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
```

---

## 10. What Recruiters Notice

| Good sign | Red flag |
|-----------|----------|
| `pyproject.toml` with metadata | No packaging config |
| CI badge on README | No CI |
| Type hints and docstrings | Untyped, undocumented code |
| Tests that pass | No tests or failing tests |
| Structured logging | `print()` everywhere |
| Narrow exception handling | Broad `except Exception` |
| `.env.example` and no secrets in repo | Hardcoded credentials |
| LICENSE file | No license |
| `__init__.py` in packages | Missing package markers |
| Shared utilities (DRY) | Copy-pasted logic |

---

## Quick Checklist

- [ ] `pyproject.toml` with project metadata and tool config
- [ ] `LICENSE` file
- [ ] `.env.example` documenting config vars
- [ ] `README.md` with badges, quick start, and architecture
- [ ] `src/` layout with `__init__.py` in every package
- [ ] Type hints (PEP 604)
- [ ] Structured logging instead of `print()`
- [ ] Narrow exception handling
- [ ] Pytest with `conftest.py` and coverage
- [ ] GitHub Actions CI (lint + test)
- [ ] Pre-commit hooks (ruff + format)
- [ ] Makefile with `lint`, `test`, `format`

---

*This project (text-ml-platform) follows these recommendations. Use it as a reference implementation.*
