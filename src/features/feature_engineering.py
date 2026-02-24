import pandas as pd

def create_features(df):
    df["lag1"] = df["Energy"].shift(1)
    df["lag7"] = df["Energy"].shift(7)
    df["rolling7"] = df["Energy"].rolling(7).mean()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df.dropna(inplace=True)
    return df
