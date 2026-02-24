# Energy Consumption Forecasting

This repository contains a simple energy consumption forecasting pipeline using SARIMA and XGBoost, with a FastAPI model serving layer and MLflow experiment tracking + model registry.

Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the synthetic dataset:

```bash
python3 create_dataset.py
```

3. Train models (logs runs to MLflow and registers models):

```bash
python3 train.py
```

4. Start the FastAPI server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

API endpoints

- `GET /` - health check
- `GET /predict` - simple query-param prediction (backwards compatible)
- `POST /predict` - JSON single-row prediction (Pydantic validated)
- `POST /predict/batch` - JSON array of inputs for batch prediction (OpenAPI example provided)
- `POST /predict/upload` - streaming CSV upload for large batch prediction (streamed parsing)
- `GET /predict_sarima` - SARIMA N-step forecast via query param
- `POST /predict_sarima` - SARIMA N-step forecast via JSON body

CSV upload details

- CSV must contain header columns: `Temperature,lag1,lag7,rolling7,dayofweek`.
- The upload endpoint streams and parses in chunks (1000 rows per chunk) to avoid loading huge files into memory.

Async CSV upload

- For very large uploads use `POST /predict/upload_async` which writes the upload to a temporary file and asynchronously parses it using `aiofiles` + `aiocsv`. This endpoint is fully non-blocking and supports large files; predictions are processed in batches to reduce memory pressure.
- Example sample CSV is available at `data/sample_batch.csv`.

Model registry

- Trained models are logged to MLflow and registered under `Energy-XGB` and `Energy-SARIMA` when possible.
- SARIMA is wrapped as an `mlflow.pyfunc` model for registry/serving.

Examples

Batch JSON example (curl):

```bash
curl -X POST http://127.0.0.1:8001/predict/batch \
 -H 'Content-Type: application/json' \
 -d '[{"Temperature":20.0,"lag1":320.0,"lag7":300.0,"rolling7":310.0,"dayofweek":2}]'
```

CSV upload example (curl):

```bash
curl -X POST http://127.0.0.1:8001/predict/upload \
 -F 'file=@data/sample_batch.csv;type=text/csv'
```

Notes

- The SARIMA pyfunc uses a pickled statsmodels results object; use the same `statsmodels` version when loading.
- For production: prefer packaging `SarimaPythonModel` in an importable module path rather than passing a live Python object to `mlflow.pyfunc.log_model`.

Want me to add OpenAPI examples for the upload endpoint request (file) or add streaming responses for very large result sets?