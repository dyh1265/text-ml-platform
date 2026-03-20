# Start demo services with GPU support (NVIDIA Container Toolkit required).
# Run from project root.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $ProjectRoot "docker" "docker-compose.yml"))) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

Write-Host "Starting infra + GPU app services from $ProjectRoot ..."
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.gpu.yml up -d --build

Write-Host ""
Write-Host "Demo services (GPU) are up. Open:"
Write-Host "  Streamlit UI:  http://localhost:8501"
Write-Host "  Predict API:   http://localhost:8000"
Write-Host ""
Write-Host "To stop: docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.gpu.yml down"
