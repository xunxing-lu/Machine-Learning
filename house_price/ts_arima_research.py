from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

df = pd.read_csv("./data/domain_properties.csv")

# For differencing test (similar to ndiffs)
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity

# Check how many differences are needed
def ndiffs_alternative(series, max_d=3):
    for d in range(max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()
        
        if check_stationarity(test_series):
            return d
    return max_d

d = ndiffs_alternative(df['price'])
print(d)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# PACF plot helps choose p → cutoff after a few significant lags
# ACF plot helps choose q → cutoff after a few significant lags

plot_acf(df['price'].diff(d).dropna())
plot_pacf(df['price'].diff(d).dropna())
plt.show()
