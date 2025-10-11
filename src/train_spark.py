# --- REAL TRAINING SCRIPT (diagnosis dropdown ready) ---
"""
Trains a Spark GBT model for hospital readmission.
- Label from 'readmitted' ("yes"/"no")
- Adds a derived diagnosis feature from diag_1 -> diag_group_id
- Logs AUC/F1 to MLflow
- Saves Spark PipelineModel and a small sklearn model for the API

Run:
  source .venv/bin/activate
  python src/train_spark.py
"""

import os
import numpy as np
import mlflow, mlflow.spark
from pyspark.sql.functions import expr, coalesce, lit


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, trim, udf, regexp_extract
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ====== Config ======
DATA_PATH    = os.getenv("DATA_PATH", "data/hospital_readmissions.csv")
RAW_LABEL    = "readmitted"     # 'yes'/'no'
LABEL_COL    = "label"          # we create this as 0/1

# Features (keep order in sync with API/UI):
FEATURES_MIN = [
    "time_in_hospital",
    "n_medications",
    "n_procedures",
    "diag_group_id",            # derived from diag_1
]

GBT_PARAMS   = dict(maxDepth=5, maxIter=50, stepSize=0.1, seed=42)

# ====== Spark ======
spark = SparkSession.builder.appName("readmission-train").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# 1) Load
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# 2) Create numeric label from RAW_LABEL (case-insensitive: yes -> 1, no -> 0)
if RAW_LABEL not in df.columns:
    raise ValueError(f"Expected column '{RAW_LABEL}' in CSV. Found: {df.columns}")

if "label" in df.columns:
    df = df.drop("label")

df = df.withColumn(RAW_LABEL, lower(trim(col(RAW_LABEL))))
df = df.withColumn(LABEL_COL, when(col(RAW_LABEL) == "yes", 1).otherwise(0).cast("int"))

# 3) Derive diag_group_id from diag_1 ICD-9 code
#    Extract leading integer from diag_1 (e.g. "250.13" -> 250)
# Safe extract: returns NULL when blank/non-numeric instead of error
df = df.withColumn(
    "diag_code",
    expr(r"try_cast(regexp_extract(cast(diag_1 as string), '^(\\d+)', 1) as int)")
)
# Optional: if you prefer a sentinel for downstream comparisons
diag_code_col = coalesce(col("diag_code"), lit(-1))



# Map ICD-9 ranges to coarse groups (0=other)
# 1: circulatory (390–459 or 785)
# 2: respiratory (460–519)
# 3: digestive (520–579)
# 4: diabetes (== 250)
# 5: injury (800–999)
# 6: musculoskeletal (710–739)
# 7: genitourinary (580–629)
# 8: neoplasms (140–239)
df = df.withColumn(
    "diag_group_id",
    when((diag_code_col >= 390) & (diag_code_col <= 459), 1)
    .when(diag_code_col == 785, 1)
    .when((diag_code_col >= 460) & (diag_code_col <= 519), 2)
    .when((diag_code_col >= 520) & (diag_code_col <= 579), 3)
    .when(diag_code_col == 250, 4)
    .when((diag_code_col >= 800) & (diag_code_col <= 999), 5)
    .when((diag_code_col >= 710) & (diag_code_col <= 739), 6)
    .when((diag_code_col >= 580) & (diag_code_col <= 629), 7)
    .when((diag_code_col >= 140) & (diag_code_col <= 239), 8)
    .otherwise(0)
    .cast("double")
)


# 4) Ensure all selected features are numeric; fill nulls
for c in FEATURES_MIN:
    if c not in df.columns:
        raise ValueError(f"Missing feature '{c}'. Available: {df.columns}")
    df = df.withColumn(c, col(c).cast("double"))
df = df.fillna(0, subset=FEATURES_MIN)

# 5) Stratified split so both classes appear in train & test
train_df = df.sampleBy(LABEL_COL, fractions={0: 0.8, 1: 0.8}, seed=42)
test_df  = df.subtract(train_df)

# (Optional) quick upsample of rare positives in TRAIN (safeguard)
pos = train_df.filter(col(LABEL_COL) == 1)
neg = train_df.filter(col(LABEL_COL) == 0)
pos_count, neg_count = pos.count(), neg.count()
if pos_count > 0 and neg_count > 0 and pos_count < neg_count:
    factor = max(int(neg_count / max(pos_count, 1)) - 1, 0)
    if factor > 0:
        up_pos = pos
        for _ in range(factor):
            up_pos = up_pos.union(pos)
        train_df = neg.union(up_pos)

# 6) Assemble & model
assembler = VectorAssembler(inputCols=FEATURES_MIN, outputCol="features")
gbt = GBTClassifier(labelCol=LABEL_COL, featuresCol="features", **GBT_PARAMS)
pipeline = Pipeline(stages=[assembler, gbt])

# 7) Fit
model = pipeline.fit(train_df)

# 8) Evaluate
preds = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
auc = evaluator.evaluate(preds)

# quick F1 at 0.5
prob1_udf = udf(lambda v: float(v[1]), "double")
preds = preds.withColumn("p1", prob1_udf(col("probability")))
preds = preds.withColumn("pred_label", when(col("p1") >= 0.5, 1).otherwise(0).cast(IntegerType()))
tp = preds.filter((col("pred_label")==1) & (col(LABEL_COL)==1)).count()
fp = preds.filter((col("pred_label")==1) & (col(LABEL_COL)==0)).count()
fn = preds.filter((col("pred_label")==0) & (col(LABEL_COL)==1)).count()
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0

# 9) Save artifacts
os.makedirs("models", exist_ok=True)
model.write().overwrite().save("models/spark_pipeline")

# 10) MLflow logging
mlflow.set_experiment("hospital_readmission")
with mlflow.start_run():
    mlflow.log_params({
        "features": ",".join(FEATURES_MIN),
        "gbt.maxDepth": GBT_PARAMS["maxDepth"],
        "gbt.maxIter":  GBT_PARAMS["maxIter"],
        "gbt.stepSize": GBT_PARAMS["stepSize"],
        "split": "stratified 80/20 by label",
        "upsample_pos_factor": int(max(int(neg_count / max(pos_count, 1)) - 1, 0)) if pos_count else 0
    })
    mlflow.log_metrics({"AUC": float(auc), "F1": float(f1)})

    try:
        mlflow.spark.log_model(model, "spark_model")
    except Exception as e:
        print("MLflow Spark log skipped:", e)

    # 11) Export a tiny sklearn model for the FastAPI
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression

        train_feat = assembler.transform(train_df).select("features", LABEL_COL)
        sample_pdf = (train_feat.sample(False, 0.5, seed=42).limit(20000)).toPandas()
        X = np.vstack(sample_pdf["features"].apply(lambda v: v.toArray()).values)
        y = sample_pdf[LABEL_COL].values.astype(int)

        if np.unique(y).size < 2:
            raise RuntimeError("sklearn export: only one class in training sample")

        sk = LogisticRegression(max_iter=300).fit(X, y)
        joblib.dump(sk, "models/sklearn_model.pkl")
        mlflow.log_artifact("models/sklearn_model.pkl")
    except Exception as e:
        print("sklearn export failed (API will use old model if present):", e)

print(f"Training complete. AUC={auc:.3f}  F1={f1:.3f}")
spark.stop()
# --- END ---
