# Hospital Readmission — ML 

This repo turns your Spark-based readmission project into a **cloud-ready ML pipeline** with:
- **FastAPI** inference (Dockerized)
- **MLflow** experiment tracking & (optional) model registry
- **CI/CD** via GitHub Actions (build + nightly retrain sketch)
- **Spark** training entrypoint (paste your code where marked)
- **Basic tests** and **schema validation**

> Paste your existing Spark logic from `bigdata_v1.py` into `src/train_spark.py` (look for `# === PASTE YOUR CODE HERE ===` blocks).

---

## Quickstart (local)

```bash
# 1) (Optional) Create/activate venv, then install
pip install -r requirements.txt

# 2) Run API locally
uvicorn api.app:app --reload --port 8080

# 3) Try a request
curl -X POST "http://127.0.0.1:8080/predict" -H "Content-Type: application/json" -d '{"time_in_hospital":3,"num_medications":12,"number_diagnoses":6}'
```

---

## Train (Spark) + Log (MLflow)

- Put your Spark pipeline in `src/train_spark.py` where indicated.
- To log with MLflow (local UI):
```bash
mlflow ui  # opens at http://127.0.0.1:5000
python src/train_spark.py
```

This will:
- Train your Spark model and save to `models/spark_pipeline/`
- Fit a lightweight sklearn model for the API and save to `models/sklearn_model.pkl`
- Log params/metrics to MLflow

---

## Docker

```bash
docker build -t readmission-api:latest .
docker run -p 8080:8080 readmission-api:latest
```

---

## Cloud Deploy (sketch)

- **GCP Cloud Run**
```bash
gcloud run deploy readmission-api --source . --region europe-west2 --platform managed
```

- **Azure App Service**
```bash
az webapp up --name readmission-api --resource-group MLAppRG --plan Basic --sku B1 --runtime "PYTHON|3.11"
```

---

## CI/CD

- See `.github/workflows/ci-cd.yml`
- On `push` to `main`: build Docker
- Nightly cron: sketch of retrain job (you can point it to Databricks or run locally and push artifacts)

---

## Tests

```bash
pytest -q
```

---


