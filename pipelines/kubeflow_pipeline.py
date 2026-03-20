"""Kubeflow pipeline definition.

This is a production-style *demo pipeline* that wires together the existing
modules in this repo:
generate (LLM client) -> bronze -> silver -> gold embeddings -> train
-> start async inference worker.

Notes:
- You must build and publish a container image that contains this repo
  (or mount the code) and has the required Python dependencies.
- For the demo, each step runs `python -m ...` commands inside the same image.
"""

from __future__ import annotations

from kfp import dsl


@dsl.pipeline(name="text-ml-platform-imdb-demo")
def imdb_demo_pipeline(
    image: str = "text-ml-platform:latest",
    llm_api_base: str = "http://vllm:8001",
    llm_model: str = "gpt2",
    generate_count: int = 50,
    bronze_batch_size: int = 100,
    embed_iceberg: bool = True,
):
    # 1) Generate synthetic reviews and publish to Kafka.
    gen = dsl.ContainerOp(
        name="generate_synthetic_reviews",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.llm.generate_reviews_client",
            "--sentiment",
            "positive",
            "--count",
            str(generate_count),
            "--split",
            "train",
            "--api-base",
            llm_api_base,
            "--model",
            llm_model,
        ],
    )

    # 2) Ingest to bronze (Kafka -> bronze JSONL).
    bronze = dsl.ContainerOp(
        name="ingest_to_bronze",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.ingestion.bronze_consumer",
            "--batch-size",
            str(bronze_batch_size),
            "--split",
            "train",
        ],
    )
    bronze.after(gen)

    # 3) Clean to silver.
    silver = dsl.ContainerOp(
        name="clean_to_silver",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.transformation.silver_job",
            "--bronze-prefix",
            "bronze/imdb/train/",
            "--silver-prefix",
            "silver/imdb/train/",
        ],
    )
    silver.after(bronze)

    # 4) Embed to gold.
    embed_args = [
        "python",
        "-m",
        "src.features.embedding_job",
        "--silver-prefix",
        "silver/imdb/train/",
        "--gold-prefix",
        "gold/imdb/train/",
    ]
    if embed_iceberg:
        embed_args.append("--iceberg")
        embed_args += ["--iceberg-namespace", "imdb", "--iceberg-table", "gold_train"]
    embed = dsl.ContainerOp(
        name="embed_to_gold",
        image=image,
        arguments=embed_args,
    )
    embed.after(silver)

    # 5) Train classifier from embeddings.
    train = dsl.ContainerOp(
        name="train_classifier",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.training.train_classifier",
            "--iceberg-identifier",
            "imdb.gold_train",
            "--train-split",
            "train",
            "--test-split",
            "test",
            "--model-out",
            "models/sentiment_logreg.joblib",
        ],
    )
    train.after(embed)

    # 6) Start async inference worker (in a real cluster you would keep it running).
    # For the demo pipeline we start it as a long-running component.
    worker_args = [
        "python",
        "-m",
        "src.inference.inference_worker",
        "--iceberg-namespace",
        "imdb",
        "--iceberg-table",
        "gold_inference",
        "--classifier-model",
        "models/sentiment_logreg.joblib",
        "--iceberg",
    ]
    worker = dsl.ContainerOp(
        name="async_inference_worker",
        image=image,
        arguments=worker_args,
    )
    worker.after(train)
