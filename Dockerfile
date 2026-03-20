FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install dependencies first for better layer caching.
# Install torch CPU-only first to avoid pulling 2+ GB of CUDA wheels (PyPI defaults to CUDA on Linux).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the project code.
COPY . /app

# Default command (overridden by kubeflow/container entrypoints).
CMD ["python", "-c", "print('text-ml-platform container ready')"]

