import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# === Load and preprocess data ===
df = pd.read_csv("https://drive.google.com/uc?export=download&id=1yPf6wVcKODwhdyZzCpv_aqZJKzmfmp0s")

# https://drive.google.com/uc?export=download&id=1yPf6wVcKODwhdyZzCpv_aqZJKzmfmp0s

# Hopefully tell whats wrong 
print("Columns:", df.columns.tolist())
print(df.head())

# === Clean and prepare datetime ===
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df['ARR_DELAY'] = df['ARR_DELAY'].fillna(0)  # Fill target NaNs with 0

# === Sort by date and group by daily average ===
df_daily = df.groupby(df['FL_DATE'].dt.date)['ARR_DELAY'].mean().reset_index()
df_daily.columns = ['Date', 'ARR_DELAY']
df_daily['Date'] = pd.to_datetime(df_daily['Date'])
df_daily = df_daily.set_index('Date')

# === Feature Engineering: create lag features ===
for lag in range(1, 8):
    df_daily[f'lag_{lag}'] = df_daily['ARR_DELAY'].shift(lag)

# Drop NaNs introduced by shifting
df_daily = df_daily.dropna()

# === Prepare features and target ===
X = df_daily.drop(columns=['ARR_DELAY'])
y = df_daily['ARR_DELAY']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# === Train model ===
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Model trained.")
print(f"📉 RMSE: {rmse:.2f}")

# === Forecasting next 365 days ===
future_steps = 365
last_known = df_daily.copy()
predictions = []

for _ in range(future_steps):
    last_row = last_known.iloc[-1]
    new_row = {}

    for lag in range(1, 8):
        new_row[f'lag_{lag}'] = last_row[f'lag_{lag-1}'] if lag > 1 else last_row['ARR_DELAY']

    new_X = pd.DataFrame([new_row])
    next_pred = model.predict(new_X)[0]
    predictions.append(next_pred)

    new_entry = new_X.iloc[0].to_dict()
    new_entry['ARR_DELAY'] = next_pred
    last_known = pd.concat([last_known, pd.DataFrame([new_entry], index=[last_known.index[-1] + pd.Timedelta(days=1)])])

# === Plot the result ===
plt.figure(figsize=(14, 6))
plt.plot(df_daily.index, df_daily['ARR_DELAY'], label="Historical")
future_index = pd.date_range(start=df_daily.index[-1] + pd.Timedelta(days=1), periods=future_steps)
plt.plot(future_index, predictions, label="Forecast", color='red')
plt.title("Arrival Delay Forecast (Next 1 Year)")
plt.xlabel("Date")
plt.ylabel("Average Arrival Delay (min)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
