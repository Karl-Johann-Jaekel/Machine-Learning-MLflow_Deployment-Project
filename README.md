# Yellow Taxi Trip Duration Estimation (ML Deployment Project)

This project trains and deploys a machine learning model to predict **trip duration** for Yellow Taxi Trips (NYC, January 2025).

## Project Goals
1. **Train** a `RandomForestRegressor` model and **track** runs with **MLflow**
2. **Deploy** the trained model as an **API** (FastAPI)
3. **Make a request** to the API and return a prediction

---

## Tech Stack
- Python 3.11+
- pandas, pyarrow (Parquet)
- scikit-learn (RandomForestRegressor + Pipeline)
- MLflow (Tracking & Model Registry-style artifact logging)
- FastAPI + Uvicorn (API Deployment)

---

## Dataset
- Source: NYC TLC Trip Record Data (Yellow Taxi)
- Year: **2025**
- Month: **01**
- Download URL (Parquet):  
  `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`

---

## Environment

Please make sure you have forked the repo and set up a new virtual environment. For this purpose you can use the following commands:

### **`macOS`**
```BASH
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
### **`WindowsOS`**
 For `PowerShell` CLI :

  ```PowerShell
  pyenv local 3.11.3
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

  For `Git-Bash` CLI :

  ```
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/Scripts/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```