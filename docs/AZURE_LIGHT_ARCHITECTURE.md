# Azure Light Architecture

A simplified deployment of the text-ml-platform that fits within **Azure free tier** or the **$200 trial credit**. Trade-offs: fewer components, smaller scale, no vLLM.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AZURE LIGHT ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────────────────────┐   │
│  │  IMDb CSV    │     │  Event Hubs     │     │  Blob Storage           │   │
│  │  (upload)    │────▶│  (Kafka-like)   │────▶│  bronze/ silver/        │   │
│  └──────────────┘     │  Free: 1M msgs  │     │  Free: 5GB, 10K reads   │   │
│                       └────────┬────────┘     └───────────┬─────────────┘   │
│                                │                          │                  │
│                                ▼                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │              Container Apps (or App Service)                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Producer   │  │  Bronze     │  │  Silver     │  │  Embedding  │  │   │
│  │  │  (batch)    │  │  Consumer   │  │  Job        │  │  Job        │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │   │
│  │  │  Finetune   │  │  Train      │  │  Predict    │                    │   │
│  │  │  BERT       │  │  Classifier │  │  API        │                    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Optional: Streamlit UI as another Container App or Static Web App           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Mapping

| Original           | Azure Light                     | Notes                                      |
|--------------------|----------------------------------|--------------------------------------------|
| Kafka              | **Event Hubs** (Kafka API)      | Free tier: 1 namespace, 1 TU, ~1M msgs/mo  |
| MinIO / S3         | **Blob Storage**                | Free: 5GB LRS, 10K reads, 1K writes/mo     |
| Spark + Iceberg    | **Blob + Parquet**              | Drop Iceberg; use Parquet files in Blob    |
| vLLM / Ollama      | **None**                        | Use IMDb producer only (no synthetic LLM)  |
| Docker Compose     | **Container Apps** or **AKS**   | Free: ~180K vCPU-s, 360K GiB-s/mo          |

---

## Free Tier Limits (Always Free)

| Service         | Limit                                           |
|-----------------|--------------------------------------------------|
| Event Hubs      | 1 namespace, 1 throughput unit, 1M msgs/month    |
| Blob Storage    | 5GB LRS, 10K read ops, 1K write ops/month        |
| Container Apps  | 180,000 vCPU-seconds, 360,000 GiB-seconds/month  |
| App Service F1  | 60 CPU min/day, 1GB RAM                          |

---

## $200 Trial (30 Days)

With the trial credit you can run a **fuller** setup:

| Service    | Use                                  | Est. cost/month      |
|------------|--------------------------------------|-----------------------|
| AKS        | Kubernetes for full Docker Compose   | ~$75 (basic nodes)    |
| Container Apps | Multi-container app              | ~$20–40               |
| Event Hubs | Kafka replacement                   | ~$10                  |
| Blob       | Object storage                      | ~$5                   |
| Cosmos DB  | (Optional) for predictions          | ~$25                  |

**Rough total:** $100–150/month → fits in $200 for ~1–2 months of light use.

---

## Code Changes Needed

1. **Event Hubs** – Use `azure-eventhub` or Kafka client with Event Hubs Kafka endpoint; connection string from Azure Portal.
2. **Blob Storage** – Use `azure-storage-blob`; adapt `src/utils/s3_client.py` to a Blob backend or use `s3fs` with Azurite / Blob Hierarchical Namespace.
3. **Drop Iceberg** – Write gold Parquet to Blob; skip Spark/Iceberg catalog.
4. **Config** – Add `AZURE_EVENTHUB_CONNECTION`, `AZURE_STORAGE_CONNECTION`, etc. in `src/config.py`.
5. **Producer** – Keep IMDb HuggingFace producer; remove or gate Ollama/vLLM path.

---

## Deployment Steps (Light – Free Tier)

### 1. Create resources

```bash
# Login
az login

# Resource group
az group create --name rg-text-ml-light --location eastus

# Event Hubs namespace (Kafka-compatible)
az eventhubs namespace create --name eh-text-ml --resource-group rg-text-ml-light \
  --location eastus --sku Basic

# Storage account (for Blob)
az storage account create --name sttextmllight --resource-group rg-text-ml-light \
  --location eastus --sku Standard_LRS
```

### 2. Build and push image

```bash
az acr create --name acrtextml --resource-group rg-text-ml-light --sku Basic
az acr login --name acrtextml
docker build -t acrtextml.azurecr.io/text-ml-platform:latest .
docker push acrtextml.azurecr.io/text-ml-platform:latest
```

### 3. Deploy to Container Apps

```bash
az containerapp env create --name env-text-ml --resource-group rg-text-ml-light --location eastus
az containerapp create --name ca-predict --resource-group rg-text-ml-light \
  --environment env-text-ml --image acrtextml.azurecr.io/text-ml-platform:latest \
  --target-port 8000 --ingress external \
  --env-vars AZURE_STORAGE_CONNECTION=@connection.txt \
             EVENTHUB_CONNECTION=@eh-connection.txt
```

---

## What You Keep vs Drop

| Kept                          | Dropped                            |
|-------------------------------|------------------------------------|
| IMDb producer (HuggingFace)   | vLLM / Ollama synthetic generation |
| Bronze → Silver → Gold flow   | Apache Iceberg (use Parquet only)  |
| BERT fine-tuning              | Spark                              |
| Embedding + classifier        | Real-time Kafka consumer (batch OK)|
| Predict API                   | Full Docker Compose parity         |
| Streamlit UI (optional)       |                                    |

---

## References

- [Event Hubs for Apache Kafka](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-for-kafka-overview)
- [Container Apps pricing](https://azure.microsoft.com/en-us/pricing/details/container-apps/)
- [Always-free Azure services](https://azure.microsoft.com/en-us/free/)
