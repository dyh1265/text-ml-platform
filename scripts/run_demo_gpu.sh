#!/usr/bin/env bash
# Start demo services with GPU support (NVIDIA Container Toolkit required).
# Run from project root.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting infra + GPU app services from $PROJECT_ROOT ..."
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.gpu.yml up -d --build

echo ""
echo "Demo services (GPU) are up. Open:"
echo "  Streamlit UI:  http://localhost:8501"
echo "  Predict API:   http://localhost:8000"
echo ""
echo "To stop: docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.gpu.yml down"
