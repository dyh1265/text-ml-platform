"""Kubeflow pipeline definition.

This is a production-style *demo pipeline* that wires together the existing
modules in this repo:

  LLM generate (pos + neg) -> bronze -> silver -> finetune BERT -> gold embeddings -> train classifier

Notes:
- You must build and publish a container image that contains this repo
  (or mount the code) and has the required Python dependencies.
- Each step runs `python -m ...` commands inside the same image (same pattern as Docker prepopulate).

**Not included:** the async inference worker is a long-running process and should be deployed as a
Kubernetes Deployment (or Docker Compose service), not as a pipeline step — otherwise the run never
completes.

**vs. Docker prepopulate:** that path uses `producer` + IMDb data and both train/test splits. This
Kubeflow demo uses the LLM client for a smaller synthetic dataset and only `gold_train`; training
uses `--test-split none` (no held-out eval unless you add test silver + embedding steps).
"""

from __future__ import annotations

from kfp import dsl


@dsl.pipeline(name="text-ml-platform-imdb-demo")
def imdb_demo_pipeline(
    image: str = "text-ml-platform:latest",
    llm_api_base: str = "http://localhost:11434/v1",
    llm_model: str = "llama3.2",
    generate_count: int = 50,
    bronze_batch_size: int = 100,
    embed_iceberg: bool = True,
    bert_output_dir: str = "models/bert_sentiment_imdb",
):
    """Parameters mirror Docker/Kubernetes service URLs; override when deploying.

    ``generate_count`` is split evenly between positive and negative synthetic reviews so
    ``train_classifier`` sees both classes (required for logistic regression).
    """
    half = generate_count // 2
    pos_count = max(1, half)
    neg_count = max(1, generate_count - pos_count)

    # 1a) Positive synthetic reviews -> Kafka
    gen_pos = dsl.ContainerOp(
        name="generate_positive_reviews",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.llm.generate_reviews_client",
            "--sentiment",
            "positive",
            "--count",
            str(pos_count),
            "--split",
            "train",
            "--api-base",
            llm_api_base,
            "--model",
            llm_model,
        ],
    )

    # 1b) Negative synthetic reviews -> Kafka (both classes needed for training)
    gen_neg = dsl.ContainerOp(
        name="generate_negative_reviews",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.llm.generate_reviews_client",
            "--sentiment",
            "negative",
            "--count",
            str(neg_count),
            "--split",
            "train",
            "--api-base",
            llm_api_base,
            "--model",
            llm_model,
        ],
    )
    gen_neg.after(gen_pos)

    total_generated = pos_count + neg_count

    # 2) Ingest to bronze (Kafka -> bronze JSONL). --limit is required or the consumer never exits.
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
            "--limit",
            str(total_generated),
        ],
    )
    bronze.after(gen_neg)

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

    # 4) Fine-tune BERT on silver (matches docker prepopulate_imdb).
    finetune = dsl.ContainerOp(
        name="finetune_bert",
        image=image,
        arguments=[
            "python",
            "-m",
            "src.training.finetune_bert",
            "--silver-train-prefix",
            "silver/imdb/train/",
            "--silver-test-prefix",
            "silver/imdb/test/",
            "--output-dir",
            bert_output_dir,
        ],
    )
    finetune.after(silver)

    # 5) Embed to gold (uses fine-tuned BERT for embeddings).
    embed_args = [
        "python",
        "-m",
        "src.features.embedding_job",
        "--silver-prefix",
        "silver/imdb/train/",
        "--bert-path",
        bert_output_dir,
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
    embed.after(finetune)

    # 6) Train classifier. No gold_test in this LLM-only path — disable test eval (see module defaults).
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
            "none",
            "--model-out",
            "models/sentiment_logreg.joblib",
        ],
    )
    train.after(embed)
