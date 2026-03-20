# Prepopulate IMDb pipeline: Kafka -> bronze -> silver -> Iceberg gold_train/gold_test -> model.
# Run from project root, with Kafka+MinIO already up (docker-compose).
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path (Join-Path $ProjectRoot "docker\\docker-compose.yml"))) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

Write-Host "Prepopulating IMDb data and model from $ProjectRoot ..."

Write-Host "1) Stream IMDb train/test into Kafka (shuffled)..."
python -m src.ingestion.producer --mode batch --split train --limit 1000
python -m src.ingestion.producer --mode batch --split test  --limit 1000

Write-Host "2) Consume Kafka -> bronze..."
python -m src.ingestion.bronze_consumer --batch-size 50 --limit 2000

Write-Host "3) Bronze -> silver (train/test)..."
python -m src.transformation.silver_job --bronze-prefix bronze/imdb/train/ --silver-prefix silver/imdb/train/
python -m src.transformation.silver_job --bronze-prefix bronze/imdb/test/  --silver-prefix silver/imdb/test/

Write-Host "4) Silver -> Iceberg gold_train/gold_test..."
python -m src.features.embedding_job --silver-prefix silver/imdb/train/ --iceberg --iceberg-namespace imdb --iceberg-table gold_train
python -m src.features.embedding_job --silver-prefix silver/imdb/test/  --iceberg --iceberg-namespace imdb --iceberg-table gold_test

Write-Host "5) Train classifier and write models/sentiment_logreg.joblib ..."
python -m src.training.train_classifier `
  --iceberg-identifier imdb.gold_train `
  --test-iceberg-identifier imdb.gold_test `
  --train-split train `
  --test-split test `
  --model-out models/sentiment_logreg.joblib

Write-Host "Done. Iceberg tables imdb.gold_train/imdb.gold_test and models/sentiment_logreg.joblib are ready."
