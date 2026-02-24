import mlflow.pyfunc
import joblib
import pandas as pd


class SarimaPythonModel(mlflow.pyfunc.PythonModel):
    """mlflow.pyfunc wrapper around a fitted SARIMA results object.

    Expects an artifact named `sarima_model.pkl` pointing to a joblib dump
    of the statsmodels results object. Predict accepts a DataFrame with an
    optional `horizon` column (integer) for each row; otherwise returns
    one-step forecasts for each row.
    """

    def load_context(self, context):
        path = context.artifacts.get("sarima_model.pkl")
        if path is None:
            raise FileNotFoundError("sarima_model.pkl not found in artifacts")
        self.res = joblib.load(path)

    import pandas as pd

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:

        # model_input may be a scalar, Series or DataFrame
        # If DataFrame contains 'horizon' column, produce last-step forecast per horizon
        if isinstance(model_input, (int, float)):
            horizon = int(model_input)
            try:
                fc = self.res.get_forecast(steps=horizon)
                return pd.Series(fc.predicted_mean.tolist())
            except Exception:
                return pd.Series(self.res.forecast(steps=horizon).tolist())

        df = pd.DataFrame(model_input)
        if "horizon" in df.columns:
            out = []
            for h in df["horizon"]:
                h = int(h)
                try:
                    fc = self.res.get_forecast(steps=h)
                    preds = fc.predicted_mean.tolist()
                except Exception:
                    preds = self.res.forecast(steps=h).tolist()
                out.append(float(preds[-1]))
            return pd.Series(out)

        # default: return 1-step forecast repeated for each input row
        try:
            fc = self.res.get_forecast(steps=1)
            val = float(fc.predicted_mean.iloc[-1])
        except Exception:
            val = float(self.res.forecast(steps=1)[-1])
        return pd.Series([val] * len(df))
