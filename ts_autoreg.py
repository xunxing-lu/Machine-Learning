from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv("./data/domain_properties.csv")
# Fix the date parsing warning - use mixed format for inconsistent date formats
df['date_sold'] = pd.to_datetime(df['date_sold'], format='mixed', dayfirst=True)
df = df[['date_sold', 'price']].dropna().sort_values('date_sold')

# Fix the deprecated 'M' frequency - use 'ME' for month end
monthly_prices = df.resample('ME', on='date_sold')['price'].mean().dropna()

# Split data
train = monthly_prices[:-3]
test = monthly_prices[-3:]

# Fit AutoReg - remove the deprecated old_names parameter
model = AutoReg(train, lags=1)
model_fit = model.fit()

# Predict next 3 months
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

# Evaluation - fix the RMSE calculation
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))  # Calculate RMSE manually

# Output
print("Forecast vs Actual:")
print(pd.DataFrame({'actual': test, 'predicted': predictions}))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")