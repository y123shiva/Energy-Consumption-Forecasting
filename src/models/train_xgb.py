import xgboost as xgb
import joblib

def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)
    joblib.dump(model, "xgb_model.pkl")
    return model
