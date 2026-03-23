"""Streamlit UI for the text-ml-platform demo.

User enters a review -> we call the prediction service -> show:
- raw review
- cleaned review
- predicted label + score
- where the cleaned text lands in 2D embedding space (PCA, t-SNE, or UMAP)
"""

from __future__ import annotations

import json
import logging
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import torch
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from pyiceberg.expressions import EqualTo
from transformers import AutoModel, AutoTokenizer

from src import config
from src.features.embedding_job import encode_batch
from src.llm.generate_reviews_client import build_prompt, openai_chat_completion
from src.transformation.cleaning import clean_text
from src.utils.iceberg_catalog import get_iceberg_catalog
from src.utils.s3_client import get_object_body, list_objects

PREDICT_API_URL = config.PREDICT_API_URL
GOLD_ICEBERG_IDENTIFIER = config.GOLD_ICEBERG_IDENTIFIER
PCA_MAX_POINTS = config.PCA_MAX_POINTS
OLLAMA_API_BASE = config.OLLAMA_API_BASE

VIZ_METHODS = ["PCA", "t-SNE", "UMAP"]


@st.cache_resource
def load_bert(model_name: str = "textattack/bert-base-uncased-SST-2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource
def load_gold_embeddings(iceberg_identifier: str, max_points: int, random_seed: int = 42):
    """Load embeddings from Iceberg to fit projection on a sampled subset."""
    catalog = get_iceberg_catalog()
    namespace, table_name = iceberg_identifier.split(".", 1)
    tbl = catalog.load_table(f"{namespace}.{table_name}")

    df = tbl.scan().to_pandas()
    if "split" in df.columns:
        df_train = df[df["split"] == "train"]
        if len(df_train) > 0:
            df = df_train

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=random_seed)

    if len(df) == 0:
        raise ValueError(f"No embeddings found in {iceberg_identifier}.")

    X = np.array(df["embedding"].tolist(), dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    return X, y


def _fit_reducer(X: np.ndarray, method: str, random_state: int = 42):
    """Fit 2D projection. Returns (reducer, coords). Reducer has transform() for PCA/UMAP; None for t-SNE."""
    if method == "PCA":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X)
        return reducer, coords
    if method == "UMAP":
        import umap  # type: ignore[import-untyped]

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X)
        return reducer, coords
    if method == "t-SNE":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X)
        return None, coords
    raise ValueError(f"Unknown method: {method}")


@st.cache_data
def get_projection(iceberg_identifier: str, max_points: int, method: str, random_seed: int = 42):
    """Load gold embeddings and fit 2D projection. Cached by (identifier, max_points, method)."""
    X, y = load_gold_embeddings(iceberg_identifier, max_points, random_seed)
    reducer, coords = _fit_reducer(X, method, random_seed)
    return reducer, coords, y


def plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    new_point: tuple[float, float] | None = None,
):
    """Plot 2D embedding with optional new point."""
    plt.figure(figsize=(8, 6))
    for lbl in [0, 1]:
        mask = labels == lbl
        plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, label=f"label={lbl}")
    if new_point is not None:
        plt.scatter([new_point[0]], [new_point[1]], s=80, marker="*", c="black", label="input")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"BERT embeddings ({method})")
    plt.legend()
    return plt.gcf()


def _stage_presence(prefix: str, request_id: str, max_objects: int = 40) -> bool:
    """Best-effort check if a JSONL object contains a given request_id."""
    try:
        objects = list_objects(prefix)
    except Exception:
        return False
    checked = 0
    for obj in objects:
        key = obj.get("Key", "")
        if not key.endswith(".jsonl"):
            continue
        checked += 1
        if checked > max_objects:
            break
        body = get_object_body(key)
        if request_id.encode("utf-8") in body:
            return True
    return False


st.set_page_config(page_title="Text ML Platform Demo", layout="wide")
if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""
# Inject generated review into the text area key before the widget is created (Streamlit disallows writing to a widget's key after creation).
if "generated_review" in st.session_state:
    st.session_state["review_input"] = st.session_state.pop("generated_review")

st.title("Text ML Platform Demo (IMDb Sentiment)")

col_left, col_right = st.columns(2)
with st.sidebar:
    st.markdown("## Iceberg snapshot overview")
    try:
        cat = get_iceberg_catalog()
        namespace, table_name = GOLD_ICEBERG_IDENTIFIER.split(".", 1)
        tbl = cat.load_table(f"{namespace}.{table_name}")
        snaps = getattr(tbl.metadata, "snapshots", [])
        st.write(f"Table: `{GOLD_ICEBERG_IDENTIFIER}`")
        st.write(f"Snapshots: {len(snaps)}")
        if snaps:
            latest = snaps[-1]
            st.write(f"Latest snapshot_id: `{latest.snapshot_id}`")
    except Exception as exc:
        st.write("Could not load Iceberg table yet.")
        st.caption(f"{str(exc)[:200]}")
        st.caption("Set env `GOLD_ICEBERG_IDENTIFIER=imdb.gold_train` (or run embedding job + create table).")
        st.caption(
            "In Docker, ensure MinIO creds are set via S3_* env vars and avoid mixing duplicate AWS/S3 credential env vars."
        )
with col_left:
    raw_review = st.text_area(
        "Enter an IMDb-like review",
        height=220,
        placeholder="Type something that describes your opinion...",
        key="review_input",
    )
    with st.expander("Generate with Ollama"):
        st.caption("Requires Ollama running locally (e.g. `ollama serve` and `ollama pull llama3.2`).")
        ollama_sentiment = st.selectbox("Sentiment", ["positive", "negative"], key="ollama_sentiment")
        ollama_model = st.text_input("Model", value="llama3.2", key="ollama_model")
        if st.button("Generate review"):
            with st.spinner("Calling Ollama..."):
                try:
                    prompt = build_prompt(ollama_sentiment)
                    text = openai_chat_completion(
                        api_base=OLLAMA_API_BASE,
                        model=ollama_model.strip() or "llama3.2",
                        prompt=prompt,
                        max_tokens=350,
                    )
                    st.session_state["generated_review"] = text.strip()
                    st.rerun()
                except Exception as e:
                    st.error(f"Ollama request failed: {e}")
with col_right:
    st.markdown("## Prediction")
    mode = st.radio("Execution mode", options=["sync", "async"], horizontal=True)
    viz_method = st.selectbox(
        "Embedding projection",
        options=VIZ_METHODS,
        index=0,
        key="viz_method",
        help="PCA: fast. t-SNE: clusters. UMAP: balance. Used for the 2D plot after prediction.",
    )
    submitted = st.button("Predict", disabled=not bool(raw_review.strip()))

if submitted:
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_raw_review", None)
    request_id = str(uuid.uuid4())
    payload = {"text": raw_review, "request_id": request_id, "split": "inference"}

    pred = None
    if mode == "sync":
        with st.spinner("Requesting prediction service..."):
            try:
                resp = requests.post(PREDICT_API_URL, json=payload, timeout=120)
                resp.raise_for_status()
                pred = resp.json()
            except Exception as exc:
                err_str = str(exc)
                st.error(f"Prediction service call failed: {exc}")
                if "refused" in err_str.lower() or "Connection" in err_str or "10061" in err_str:
                    st.info(
                        "Start the prediction service in a terminal (from project root): "
                        "`python -m src.inference.predict_service --port 8002`"
                    )
    else:
        kafka_ok = False

        with st.spinner("Sending inference request to Kafka..."):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                )
                msg = {
                    "id": request_id,
                    "text": raw_review,
                    "label": 0,
                    "split": "inference",
                    "request_id": request_id,
                }
                producer.send(config.KAFKA_INFERENCE_TOPIC, msg)
                producer.flush()
                producer.close()
                kafka_ok = True
            except NoBrokersAvailable as exc:
                st.error(f"Kafka broker not available: {exc}")
            except Exception as exc:
                st.error(f"Failed to send Kafka message: {exc}")

        if kafka_ok:
            st.success(
                f"**Queued** — message sent to Kafka topic `{config.KAFKA_INFERENCE_TOPIC}` "
                f"(request_id `{request_id[:8]}…`)"
            )

            with st.status("Async pipeline — waiting for inference worker", expanded=True) as status_box:
                status_box.write(
                    "The worker runs: **bronze → silver → BERT → gold_inference → predictions**. "
                    "Polling the predictions Iceberg table for results…"
                )
                progress_markdown = st.empty()

                catalog = get_iceberg_catalog()
                predictions_table = "imdb.predictions"
                max_wait_s = config.ASYNC_PREDICTION_TIMEOUT_S
                start = time.time()
                poll_count = 0

                while time.time() - start < max_wait_s:
                    elapsed = time.time() - start
                    poll_count += 1
                    try:
                        tbl = catalog.load_table(predictions_table)
                        hit = tbl.scan(row_filter=EqualTo("request_id", request_id)).to_pandas()
                        if len(hit) > 0:
                            row = hit.iloc[-1]
                            progress_markdown.markdown(f"**Elapsed:** {elapsed:.1f}s — **prediction found** ✅")
                            pred = {
                                "request_id": row["request_id"],
                                "pred_label": int(row["pred_label"]),
                                "pred_score": float(row["pred_score"]),
                                "cleaned_text": row["cleaned_text"],
                            }
                            break
                    except Exception as e:
                        logging.getLogger(__name__).debug("Async poll attempt failed (will retry): %s", e)

                    progress_markdown.markdown(
                        f"**Elapsed:** {elapsed:.1f}s / {max_wait_s}s — polling `{predictions_table}` "
                        f"(attempt {poll_count})…"
                    )
                    if elapsed < 5:
                        time.sleep(0.5)
                    elif elapsed < 15:
                        time.sleep(1.0)
                    else:
                        time.sleep(2.0)

            if pred is not None:
                status_box.update(label="Async pipeline — **complete**", state="complete", expanded=False)
            else:
                status_box.update(label="Async pipeline — **timed out**", state="error", expanded=True)
                progress_markdown.empty()
                st.error("Timed out waiting for prediction. Ensure inference_worker is running.")
                st.info(
                    "Start the async inference worker in a terminal (from project root, after training): "
                    "`python -m src.inference.inference_worker --classifier-model models/sentiment_logreg.joblib --iceberg --iceberg-namespace imdb --iceberg-table gold_inference`"
                )

    if pred is not None:
        st.session_state["last_prediction"] = pred
        st.session_state["last_raw_review"] = raw_review

    pred = st.session_state.get("last_prediction")
    if pred is not None:
        cleaned = pred.get("cleaned_text", "")
        pred_label = int(pred.get("pred_label"))
        pred_score = float(pred.get("pred_score"))
        request_id = pred.get("request_id", "")
        raw_review_for_embed = st.session_state.get("last_raw_review", raw_review)

        st.success(f"Prediction: {'positive' if pred_label == 1 else 'negative'} (score={pred_score:.3f})")
        st.write(f"request_id: `{pred.get('request_id')}`")

        st.markdown("### Cleaned text")
        st.code(cleaned[:2000])

        st.markdown("### Request-level trace (best-effort)")
        st.caption(
            "In **sync** mode the request never goes through Kafka/bronze/silver, so both stay false. In **async** mode (with inference_worker running) the worker writes to bronze and silver, so they become true after processing."
        )
        bronze_prefix = f"{config.BRONZE_PREFIX}imdb/inference/"
        silver_prefix = f"{config.SILVER_PREFIX}imdb/inference/"
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Bronze record found: `{_stage_presence(bronze_prefix, request_id)}`")
        with col2:
            st.write(f"Silver record found: `{_stage_presence(silver_prefix, request_id)}`")

        # Visualize in 2D embedding space (uses projection selected above).
        st.markdown("### Embedding visualization")
        model, tokenizer, device = load_bert()
        cleaned_for_embed = cleaned if cleaned else clean_text(raw_review_for_embed)
        new_emb = None
        if cleaned_for_embed:
            with torch.no_grad():
                enc = encode_batch(model, tokenizer, [cleaned_for_embed], torch.device(device))
            new_emb = enc[0].cpu().numpy().astype(np.float32).reshape(1, -1)

        max_pts = min(500, PCA_MAX_POINTS) if viz_method == "t-SNE" else PCA_MAX_POINTS
        if viz_method == "t-SNE" and new_emb is not None:
            # t-SNE has no transform(); refit including new point.
            with st.spinner("Fitting t-SNE (including your review; this may take a moment)..."):
                X_gold, y_gold = load_gold_embeddings(GOLD_ICEBERG_IDENTIFIER, max_pts)
                X_combined = np.vstack([X_gold, new_emb])
                _, coords_combined = _fit_reducer(X_combined, "t-SNE")
                coords = coords_combined[:-1]
                new_point = tuple(coords_combined[-1])
        else:
            with st.spinner(f"Loading gold embeddings + fitting {viz_method}..."):
                reducer, coords, y_gold = get_projection(GOLD_ICEBERG_IDENTIFIER, max_pts, viz_method)
            new_point = tuple(reducer.transform(new_emb)[0]) if new_emb is not None and reducer is not None else None

        fig = plot_embeddings(coords, y_gold, viz_method, new_point=new_point)
        st.pyplot(fig)
        if viz_method == "t-SNE":
            st.caption("t-SNE uses up to 500 points for speed. Change to PCA or UMAP for full dataset.")
else:
    st.info("Enter a review and click Predict.")
