from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import io
import tempfile
import pathlib

# async CSV parsing
import aiofiles
import aiocsv

app = FastAPI(title="Energy Forecast API")

# load model at startup (expects xgb_model.pkl present in repo root)
model = joblib.load("xgb_model.pkl")
# try to load SARIMA results object if present
try:
    sarima_model = joblib.load("sarima_model.pkl")
except Exception:
    sarima_model = None


class XGBInput(BaseModel):
    Temperature: float = Field(..., description="Current temperature")
    lag1: float = Field(..., description="Energy at t-1")
    lag7: float = Field(..., description="Energy at t-7")
    rolling7: float = Field(..., description="7-day rolling mean")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")

    class Config:
        schema_extra = {
            "example": {
                "Temperature": 20.0,
                "lag1": 320.0,
                "lag7": 300.0,
                "rolling7": 310.0,
                "dayofweek": 2,
            }
        }


class SarimaInput(BaseModel):
    horizon: int = Field(1, ge=1, description="Forecast horizon in steps")

    class Config:
        schema_extra = {"example": {"horizon": 3}}


@app.get("/")
def home():
    return {"message": "Energy Forecast API is running 🚀"}


@app.get("/predict")
def predict(lag1: float, lag7: float, rolling7: float, dayofweek: int, temp: float):
    # Backwards-compatible GET endpoint using simple query params
    data = {
        "Temperature": [temp],
        "lag1": [lag1],
        "lag7": [lag7],
        "rolling7": [rolling7],
        "dayofweek": [dayofweek],
    }
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    return {"forecast_energy": float(pred)}


@app.post("/predict")
def predict_post(payload: XGBInput):
    # POST endpoint with Pydantic validation
    df = pd.DataFrame([payload.dict()])
    pred = model.predict(df)[0]
    return {"forecast_energy": float(pred)}


@app.post("/predict/batch")
def predict_batch(
    payloads: list[XGBInput] = Body(
        ...,
        example=[
            {
                "Temperature": 20.0,
                "lag1": 320.0,
                "lag7": 300.0,
                "rolling7": 310.0,
                "dayofweek": 2,
            },
            {
                "Temperature": 21.0,
                "lag1": 321.0,
                "lag7": 301.0,
                "rolling7": 311.0,
                "dayofweek": 3,
            },
        ],
    )
):
    """Accept a JSON array of inputs for batch prediction."""
    df = pd.DataFrame([p.dict() for p in payloads])
    preds = model.predict(df).tolist()
    return {"forecasts": [float(x) for x in preds]}


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Accept a CSV file upload (streamed) containing the feature columns for batch prediction.

    CSV must include: Temperature, lag1, lag7, rolling7, dayofweek

    Example curl:
    curl -X POST http://127.0.0.1:8001/predict/upload -F 'file=@data/sample_batch.csv;type=text/csv'
    """
    # Stream-parse CSV in chunks to support large files without loading whole file
    expected = ["Temperature", "lag1", "lag7", "rolling7", "dayofweek"]
    try:
        # wrap the underlying file (binary) into a text stream for pandas
        file.file.seek(0)
        stream = io.TextIOWrapper(file.file, encoding="utf-8")
        preds = []
        for chunk in pd.read_csv(stream, chunksize=1000):
            if not all(c in chunk.columns for c in expected):
                raise HTTPException(status_code=400, detail=f"CSV must contain columns: {expected}")
            chunk = chunk[expected]
            chunk_preds = model.predict(chunk).tolist()
            preds.extend(chunk_preds)
    except HTTPException:
        raise
    except Exception:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"unable to parse/process CSV file: {tb}")
    return {"forecasts": [float(x) for x in preds]}



@app.post("/predict/upload_async", summary="Async CSV upload (streamed)", description="Upload a CSV file and process it asynchronously using aiofiles + aiocsv. See `data/sample_batch.csv` for an example CSV file.")
async def predict_upload_async(file: UploadFile = File(..., description="CSV file. Columns: Temperature, lag1, lag7, rolling7, dayofweek")):
    """Accept a CSV file upload and stream-parse it asynchronously for large files.

    The endpoint writes the incoming upload to a temporary file and uses `aiocsv.AsyncDictReader`
    to iterate rows without loading the whole file into memory. Predictions are batched for performance.
    """
    expected = ["Temperature", "lag1", "lag7", "rolling7", "dayofweek"]
    # write upload to a temp file
    try:
        suffix = pathlib.Path(file.filename).suffix or ".csv"
        async with aiofiles.tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            tmp_name = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await tmp.write(chunk)
        preds = []
        # now open the temp file for async reading and parse with aiocsv
        async with aiofiles.open(tmp_name, mode="r", encoding="utf-8") as afp:
            reader = aiocsv.AsyncDictReader(afp)
            batch_rows = []
            batch_size = 1024
            async for row in reader:
                # aiocsv returns all values as strings; coerce types
                try:
                    row_conv = {
                        "Temperature": float(row.get("Temperature", "")),
                        "lag1": float(row.get("lag1", "")),
                        "lag7": float(row.get("lag7", "")),
                        "rolling7": float(row.get("rolling7", "")),
                        "dayofweek": int(row.get("dayofweek", "")),
                    }
                except Exception:
                    raise HTTPException(status_code=400, detail="CSV rows must contain columns: Temperature, lag1, lag7, rolling7, dayofweek with numeric values")
                batch_rows.append(row_conv)
                if len(batch_rows) >= batch_size:
                    df_chunk = pd.DataFrame(batch_rows)
                    preds.extend(model.predict(df_chunk).tolist())
                    batch_rows = []
            # final batch
            if batch_rows:
                df_chunk = pd.DataFrame(batch_rows)
                preds.extend(model.predict(df_chunk).tolist())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"async upload error: {exc}")
    finally:
        # cleanup temp file if exists
        try:
            pathlib.Path(tmp_name).unlink()
        except Exception:
            pass
    return {"forecasts": [float(x) for x in preds]}


@app.get("/predict_sarima")
def predict_sarima(horizon: int = 1):
    """Return an N-step SARIMA forecast using the saved SARIMA results object.

    Example: `/predict_sarima?horizon=3` -> returns 3 future steps.
    """
    if sarima_model is None:
        raise HTTPException(status_code=404, detail="SARIMA model not found; train and save sarima_model.pkl")
    try:
        # preferred API when available
        fc = sarima_model.get_forecast(steps=horizon)
        preds = fc.predicted_mean.tolist()
    except Exception:
        preds = sarima_model.forecast(steps=horizon).tolist()
    return {"sarima_forecast": [float(x) for x in preds]}


@app.post("/predict_sarima")
def predict_sarima_post(payload: SarimaInput):
    if sarima_model is None:
        raise HTTPException(status_code=404, detail="SARIMA model not found; train and save sarima_model.pkl")
    horizon = int(payload.horizon)
    try:
        fc = sarima_model.get_forecast(steps=horizon)
        preds = fc.predicted_mean.tolist()
    except Exception:
        preds = sarima_model.forecast(steps=horizon).tolist()
    return {"sarima_forecast": [float(x) for x in preds]}
