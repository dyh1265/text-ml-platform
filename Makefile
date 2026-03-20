ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

.PHONY: lint format test test-cov test-all docker-up docker-down docker-rebuild prepopulate \
	docker-up-cpu docker-down-cpu docker-rebuild-cpu prepopulate-cpu clean clean-minio

lint:
	python -m ruff check src/ tests/
	python -m ruff format --check src/ tests/

format:
	python -m ruff check --fix src/ tests/
	python -m ruff format src/ tests/

test:
	python -m pytest tests/ -v --ignore=tests/test_gold_iceberg_table.py

test-cov:
	python -m pytest tests/ -v --ignore=tests/test_gold_iceberg_table.py --cov=src --cov-report=term-missing

test-all:
	python -m pytest tests/ -v

COMPOSE_GPU = docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.gpu.yml
COMPOSE_CPU = docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml

docker-up:
	$(COMPOSE_GPU) up -d --build

docker-down:
	$(COMPOSE_GPU) down

docker-rebuild:
	$(COMPOSE_GPU) down
	$(COMPOSE_GPU) up -d --build

docker-rebuild-clean:
	$(COMPOSE_GPU) down
	docker builder prune -af
	$(COMPOSE_GPU) build --no-cache
	$(COMPOSE_GPU) up -d

prepopulate:
	$(COMPOSE_GPU) up --abort-on-container-exit prepopulate_imdb

docker-up-cpu:
	$(COMPOSE_CPU) up -d --build

docker-down-cpu:
	$(COMPOSE_CPU) down

docker-rebuild-cpu:
	$(COMPOSE_CPU) down
	$(COMPOSE_CPU) up -d --build

prepopulate-cpu:
	$(COMPOSE_CPU) run --rm prepopulate_imdb

clean:
	python -c "from pathlib import Path; import shutil; patterns=['__pycache__','.pytest_cache','.ruff_cache','.mypy_cache']; [shutil.rmtree(p, ignore_errors=True) for pat in patterns for p in Path('.').rglob(pat) if p.is_dir()]; [shutil.rmtree(p, ignore_errors=True) for p in [Path('dist'), Path('build')] if p.exists()]; [shutil.rmtree(p, ignore_errors=True) for p in Path('.').glob('*.egg-info') if p.is_dir()]"

clean-minio: docker-down
ifeq ($(OS),Windows_NT)
	New-Item -ItemType Directory -Force -Path empty_temp | Out-Null
	-robocopy empty_temp docker\minio-data /MIR
	Remove-Item -Recurse -Force empty_temp -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force docker\minio-data -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force iceberg_catalog -ErrorAction SilentlyContinue
else
	rm -rf docker/minio-data iceberg_catalog
endif
