import pandas as pd
import mlflow
from src.features.feature_engineering import create_features
from src.models.train_xgb import train_xgb
from src.utils.metrics import evaluate
from src.models.train_sarima import train_sarima
import joblib
from src.utils.model_registry import register_models_from_run
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from src.models.sarima_pyfunc import SarimaPythonModel

def run_training():

    df = pd.read_csv("data/energy_data.csv", parse_dates=["Date"])
    df = create_features(df)

    X = df.drop(columns=["Energy", "Date"])
    y = df["Energy"]

    split = int(len(df)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    with mlflow.start_run():

        model = train_xgb(X_train, y_train)

        preds = model.predict(X_test)

        mae, rmse, mape = evaluate(y_test, preds)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)

        print("MAE:", mae)

        # Attach an explicit signature + input example for XGBoost model
        try:
            # Build a typed input schema from the training feature names
            input_cols = []
            for c in X_train.columns:
                if c == "dayofweek":
                    input_cols.append(ColSpec("long", c))
                else:
                    input_cols.append(ColSpec("double", c))
            input_schema_xgb = Schema(input_cols)
            output_schema_xgb = Schema([ColSpec("double", "forecast")])
            sig_xgb = ModelSignature(inputs=input_schema_xgb, outputs=output_schema_xgb)
            input_example_xgb = X_test.head(1)
            mlflow.sklearn.log_model(
                model,
                artifact_path="xgb_model",
                signature=sig_xgb,
                input_example=input_example_xgb,
            )
        except Exception:
            # fallback: best-effort logging without signature
            try:
                mlflow.sklearn.log_model(model, name="xgb_model")

            except Exception:
                pass

        # Train SARIMA on the training energy series and save artifact
        series = df.set_index("Date")["Energy"]
        series_train = series[:split]
        sarima_res = train_sarima(series_train)
        # persist SARIMA results object
        joblib.dump(sarima_res, "sarima_model.pkl")

        # Log raw pkl artifacts as well
        try:
            mlflow.log_artifact("xgb_model.pkl")
        except Exception:
            pass
        try:
            mlflow.log_artifact("sarima_model.pkl")
        except Exception:
            pass

        # Prepare explicit signature and input example for SARIMA pyfunc
        try:
            input_example_sarima = pd.DataFrame({"horizon": [1]})
            # produce an example output for horizon=1
            try:
                out_example = sarima_res.get_forecast(steps=1).predicted_mean.tolist()
            except Exception:
                out_example = sarima_res.forecast(steps=1).tolist()
            input_schema_sarima = Schema([ColSpec("long", "horizon")])
            output_schema_sarima = Schema([ColSpec("double", "forecast")])
            sig_sarima = ModelSignature(inputs=input_schema_sarima, outputs=output_schema_sarima)
        except Exception:
            input_example_sarima = pd.DataFrame({"horizon": [1]})
            sig_sarima = None

        # Log SARIMA as a pyfunc model so it can be registered and served
        try:
            kwargs = {}
            if sig_sarima is not None:
                kwargs["signature"] = sig_sarima
                kwargs["input_example"] = input_example_sarima
            mlflow.pyfunc.log_model(name="sarima_model",
                python_model=SarimaPythonModel(),
                artifacts={"sarima_model.pkl": "sarima_model.pkl"},
                **kwargs,
            )
        except Exception:
            pass

        # Attempt to register MLflow model flavors in the Model Registry
        run_id = mlflow.active_run().info.run_id
        registrations = {
            # mlflow.sklearn.log_model used artifact path 'xgb_model'
            "Energy-XGB": "xgb_model",
            # mlflow.pyfunc.log_model used artifact path 'sarima_model'
            "Energy-SARIMA": "sarima_model",
        }
        try:
            register_models_from_run(run_id, registrations)
        except Exception:
            # registry might not be available in this tracking server/setup
            pass
