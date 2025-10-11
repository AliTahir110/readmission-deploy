#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Training (Spark) + MLflow logging..."
python src/train_spark.py

echo "[2/3] Starting API locally on :8080"
uvicorn api.app:app --host 0.0.0.0 --port 8080
