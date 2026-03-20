#!/usr/bin/env bash
# Start all demo services (Kafka, MinIO, bronze consumer, predict service, inference worker, Streamlit).
# Run from project root. Requires Docker. Train the classifier first so models/sentiment_logreg.joblib exists.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting infra + app services from $PROJECT_ROOT ..."
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml up -d --build

echo ""
echo "Demo services are up. Open:"
echo "  Streamlit UI:  http://localhost:8501"
echo "  Predict API:   http://localhost:8000"
echo ""
echo "To stop: docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml down"
