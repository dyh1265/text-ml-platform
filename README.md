# text-ml-platform

Text ML platform for ingestion, transformation, feature engineering, and training.

## Structure

- **docker/** – Docker Compose and container config
- **src/** – Application code (config, ingestion, transformation, features, training, utils)
- **pipelines/** – Kubeflow pipeline definitions
- **tests/** – Unit and integration tests

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Kafka + IMDb ingestion

1. Make sure Docker is running, then from the project root start the stack:

   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

2. Stream a small sample of IMDb Movie Reviews into Kafka:

   ```bash
   python -m src.ingestion.producer --mode batch --limit 100
   ```

   This downloads the IMDb dataset via the `datasets` library and sends reviews to the `imdb-reviews` topic.

3. In another terminal, run the bronze consumer to land the data in the bronze layer (MinIO/S3):

   ```bash
   python -m src.ingestion.bronze_consumer --batch-size 50
   ```

   JSONL files will be written under the `bronze/imdb/` prefix in the bucket configured in `src/config.py`.

4. Run the silver job to clean and transform bronze data:

   ```bash
   python -m src.transformation.silver_job
   ```

   This reads JSONL from `bronze/imdb/`, applies text cleaning, deduplication, and writes to `silver/imdb/`.

5. Run the embedding job to produce gold Parquet (BERT-encoded):

   ```bash
   python -m src.features.embedding_job
   ```

   This reads silver JSONL, encodes text with BERT, and writes Parquet to `gold/imdb/`.

**Gold as Iceberg:** Use `--iceberg` to write to an Iceberg table (ACID, time travel, schema evolution):

```bash
python -m src.features.embedding_job --iceberg
```

See `iceberg_gold_tutorial.ipynb` for why Iceberg and how to use it.

For tutorials, see `kafka_tutorial.ipynb`, `stream_to_bronze_tutorial.ipynb`, `silver_layer_tutorial.ipynb`, `gold_layer_tutorial.ipynb`, and `iceberg_gold_tutorial.ipynb`.
