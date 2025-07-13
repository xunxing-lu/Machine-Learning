from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv("./data/domain_properties.csv")
df['date_sold'] = pd.to_datetime(df['date_sold'], format='mixed', dayfirst=True)
df = df[['date_sold', 'price']].dropna().sort_values('date_sold')

# Create monthly time series
monthly_prices = df.resample('ME', on='date_sold')['price'].mean().dropna()

# Split data
train = monthly_prices[:-3]
test = monthly_prices[-3:]

print(f"Training data points: {len(train)}")
print(f"Test data points: {len(test)}")

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("Series is stationary")
        return True
    else:
        print("Series is non-stationary")
        return False

print("\nChecking stationarity of training data:")
is_stationary = check_stationarity(train)

# If not stationary, difference the data
if not is_stationary:
    train_diff = train.diff().dropna()
    print("\nChecking stationarity after differencing:")
    check_stationarity(train_diff)

# Method 1: Manual ARIMA with specific parameters
print("\n=== Method 1: Manual ARIMA(1,1,1) ===")
try:
    # ARIMA(p,d,q) where p=AR order, d=differencing, q=MA order
    model_manual = ARIMA(train, order=(1,1,1))
    model_fit_manual = model_manual.fit()
    
    print(model_fit_manual.summary())
    
    # Forecast
    forecast_manual = model_fit_manual.forecast(steps=len(test))
    
    # Evaluation
    mae_manual = mean_absolute_error(test, forecast_manual)
    rmse_manual = np.sqrt(mean_squared_error(test, forecast_manual))
    
    print(f"\nManual ARIMA(1,1,1) Results:")
    print(f"MAE: {mae_manual:.2f}, RMSE: {rmse_manual:.2f}")
    
    # Show results
    results_manual = pd.DataFrame({
        'actual': test,
        'predicted': forecast_manual
    })
    print("\nForecast vs Actual:")
    print(results_manual)
    
except Exception as e:
    print(f"Error with manual ARIMA: {e}")

# Method 2: Auto ARIMA (requires pmdarima package)
print("\n=== Method 2: Auto ARIMA ===")
try:
    from pmdarima import auto_arima
    
    # Auto ARIMA to find best parameters
    auto_model = auto_arima(train, 
                           start_p=0, start_q=0,
                           max_p=3, max_q=3,
                           seasonal=False,
                           stepwise=True,
                           suppress_warnings=True,
                           error_action='ignore',
                           max_iter=10,
                           out_of_sample_size=len(test))
    
    print(f"Best ARIMA order: {auto_model.order}")
    print(auto_model.summary())
    
    # Forecast
    forecast_auto = auto_model.predict(n_periods=len(test))
    
    # Evaluation
    mae_auto = mean_absolute_error(test, forecast_auto)
    rmse_auto = np.sqrt(mean_squared_error(test, forecast_auto))
    
    print(f"\nAuto ARIMA Results:")
    print(f"MAE: {mae_auto:.2f}, RMSE: {rmse_auto:.2f}")
    
    # Show results
    results_auto = pd.DataFrame({
        'actual': test,
        'predicted': forecast_auto
    })
    print("\nForecast vs Actual:")
    print(results_auto)
    
except ImportError:
    print("pmdarima not installed. To use auto_arima, install with: pip install pmdarima")
except Exception as e:
    print(f"Error with auto ARIMA: {e}")

# Method 3: Grid search for best ARIMA parameters
print("\n=== Method 3: Grid Search ARIMA ===")
def evaluate_arima_model(X, arima_order):
    train_size = len(X) - 3
    train, test = X[:train_size], X[train_size:]
    try:
        model = ARIMA(train, order=arima_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        error = mean_absolute_error(test, forecast)
        return error
    except:
        return float('inf')

# Grid search
p_values = range(0, 3)
d_values = range(0, 2)  
q_values = range(0, 3)

best_score, best_cfg = float('inf'), None
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                mae = evaluate_arima_model(monthly_prices, order)
                if mae < best_score:
                    best_score, best_cfg = mae, order
                print(f'ARIMA{order} MAE={mae:.3f}')
            except:
                continue

print(f'\nBest ARIMA{best_cfg} MAE={best_score:.3f}')

# Fit the best model
if best_cfg:
    best_model = ARIMA(train, order=best_cfg)
    best_model_fit = best_model.fit()
    best_forecast = best_model_fit.forecast(steps=len(test))
    
    best_mae = mean_absolute_error(test, best_forecast)
    best_rmse = np.sqrt(mean_squared_error(test, best_forecast))
    
    print(f"\nBest ARIMA{best_cfg} Results:")
    print(f"MAE: {best_mae:.2f}, RMSE: {best_rmse:.2f}")
    
    # Show results
    results_best = pd.DataFrame({
        'actual': test,
        'predicted': best_forecast
    })
    print("\nForecast vs Actual:")
    print(results_best)

# Plotting the results (optional)
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values, label='Training Data', color='blue')
plt.plot(test.index, test.values, label='Actual', color='red', marker='o')

if 'forecast_manual' in locals():
    plt.plot(test.index, forecast_manual, label='ARIMA(1,1,1)', color='green', marker='s')

if 'best_forecast' in locals():
    plt.plot(test.index, best_forecast, label=f'Best ARIMA{best_cfg}', color='orange', marker='^')

plt.title('ARIMA Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()