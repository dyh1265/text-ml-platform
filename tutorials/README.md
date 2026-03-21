# Tutorials

Jupyter notebooks for learning the text-ml-platform pipeline step by step.

**Run from project root** so `src` imports and `docker` paths work:

```bash
# From project root
jupyter notebook tutorials/kafka_tutorial.ipynb
# or
jupyter lab tutorials/
```

| Notebook | Description |
|----------|-------------|
| `kafka_tutorial.ipynb` | Kafka basics and IMDb producer |
| `stream_to_bronze_tutorial.ipynb` | Producer → Kafka → Bronze consumer |
| `silver_layer_tutorial.ipynb` | Bronze → Silver (cleaning, dedup) |
| `gold_layer_tutorial.ipynb` | Silver → BERT embeddings → Parquet |
| `iceberg_gold_tutorial.ipynb` | Iceberg schema evolution, time travel |
| `production_demo_tutorial.ipynb` | Full flow: Kafka → Bronze/Silver/Gold → Training → UI |
