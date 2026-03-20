# Start all demo services (Kafka, MinIO, bronze consumer, predict service, inference worker, Streamlit).
# Run from project root. Requires Docker. Train the classifier first so models/sentiment_logreg.joblib exists.

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $ProjectRoot "docker" "docker-compose.yml"))) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

Write-Host "Starting infra + app services from $ProjectRoot ..."
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml up -d --build

Write-Host ""
Write-Host "Demo services are up. Open:"
Write-Host "  Streamlit UI:  http://localhost:8501"
Write-Host "  Predict API:   http://localhost:8000"
Write-Host ""
Write-Host "To stop: docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml down"
