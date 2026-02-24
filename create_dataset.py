import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

temp = 25 + 8*np.sin(np.arange(len(dates))/30) + np.random.normal(0,2,len(dates))
base = 300 + 60*np.sin(np.arange(len(dates))/7)

energy = base + temp*5 + np.random.normal(0,10,len(dates))

df = pd.DataFrame({
    "Date": dates,
    "Temperature": temp,
    "Energy": energy
})

df.to_csv("data/energy_data.csv", index=False)

print("Saved → data/energy_data.csv")
