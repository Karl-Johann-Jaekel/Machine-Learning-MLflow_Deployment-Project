from __future__ import annotations

from pathlib import Path
import os
import urllib.request

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
#  Config
# ============================================================
TRACKING_URI = "http://84.247.128.24:5000" # tells MLflow which tracking server to log to (own remote server)
EXPERIMENT_NAME = "yellow-taxi-time-estimation" # all runs will be grouped under this experiment.

URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet" # data source

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "yellow_tripdata_2025-01.parquet"

# optional quicktest
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "200000")) 


# ============================================================
#  Download Data
# ============================================================
def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    print(f"Downloading dataset from:\n  {url}\n-> {path}")
    urllib.request.urlretrieve(url, path)
    print("Download complete.")


# ============================================================
#  Load + Feature Engineering
# ============================================================
def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    # Tipp: Wenn du hier einen ImportError zu "pyarrow" bekommst:
    # pip install pyarrow
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "RatecodeID",
        "payment_type",
        "VendorID",
        "store_and_fwd_flag",
    ]

    df = pd.read_parquet(path, columns=columns)

    # Optional sampling for fast Iterationen
    if SAMPLE_SIZE and SAMPLE_SIZE > 0 and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # Datetime parsing 
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    # Target: duration in secounds
    df["duration_sec"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()

    # Basic cleaning
    df = df.dropna(subset=["duration_sec", "trip_distance", "passenger_count", "PULocationID", "DOLocationID"])
    df = df[(df["duration_sec"] > 30) & (df["duration_sec"] < 3 * 60 * 60)]  
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 200)]
    df = df[(df["passenger_count"] >= 0) & (df["passenger_count"] <= 8)]

    # Time-Features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour.astype("int16")
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday.astype("int16")

    # Features / Target
    feature_cols_num = ["passenger_count", "trip_distance", "pickup_hour", "pickup_weekday"]
    feature_cols_cat = ["PULocationID", "DOLocationID", "RatecodeID", "payment_type", "VendorID", "store_and_fwd_flag"]

    X = df[feature_cols_num + feature_cols_cat]
    y = df["duration_sec"].astype("float32")

    return X, y


# ============================================================
#  Train + Evaluate + Log to MLflow
# ============================================================
def main() -> None:
    download_if_missing(URL, DATA_PATH)

    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = ["passenger_count", "trip_distance", "pickup_hour", "pickup_weekday"]
    categorical_features = ["PULocationID", "DOLocationID", "RatecodeID", "payment_type", "VendorID", "store_and_fwd_flag"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    n_estimators = 200
    max_depth = 20

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # MLflow Setup
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # Log params + metrics
        mlflow.log_param("dataset_url", URL)
        mlflow.log_param("sample_size", int(len(X)))
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("rmse", rmse)

        # Log model (Pipeline inklusive OneHotEncoder + RandomForest)
        mlflow.sklearn.log_model(pipeline, "model")

        print("Done. RMSE:", rmse)


if __name__ == "__main__":
    main()
