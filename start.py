import numpy as np
import pandas as pd
from statistics import mean
from math import sqrt

remote_repo = 'https://raw.githubusercontent.com/drnesr/WaterConsumption/master/data/SampleData.csv'

df = pd.read_csv(remote_repo, skipinitialspace=True, parse_dates=['Date'])

df = df.set_index('Date')

# add missing values
# df = df.assign(FillMean=df.target.fillna(df.target.mean()))
# df = df.assign(FillMedian=df.target.fillna(df.target.median()))
# df['Mode'] = df.target.fillna(df.target.mode())

df['RollMean'] = df.target.fillna(df.target.rolling(24, min_periods=1).mean())
df['RollMedian'] = df.target.fillna(df.target.rolling(24, min_periods=1).median())

df['Linear'] = df.target.fillna(df.target.interpolate(method='linear', limit_direction='forward'))

df = df.assign(Linear=df.target.fillna(
    df.target.interpolate(method='linear', limit_direction='forward')))

df['Quadratic'] = df.target.fillna(df.target.interpolate(
    method='quadratic', limit_direction='forward'))


# stats

# df.dropna(inplace=True)

x = np.array(df.reference)
y = np.array(df.Linear)


def best_fit_line(x, y):
    m = (((mean(x) * mean(y)) - mean(x * y)) /
         (((mean(x))**2) - (mean((x**2)))
          ))
    b = mean(y) - m * mean(x)
    return m, b


def regression_line(m, b, x):
    return np.array(list(m * i + b for i in x))


def correlation_coefficient(x, y):
    r = (sum((x - mean(x)) * (y - mean(y))) /
         (sqrt(sum((x - mean(x))**2)) * sqrt(sum((y - mean(y))**2))))
    return r


r = correlation_coefficient(x, y)

r20 = correlation_coefficient(np.array(df.reference), np.array(df.RollMean))
r30 = correlation_coefficient(np.array(df.reference), np.array(df.RollMedian))
r40 = correlation_coefficient(np.array(df.reference), np.array(df.Quadratic))

# results

m, b = best_fit_line(x, y)
regression_line = regression_line(m, b, x)

print(df[:5])
print('linear', r)
print('rollmean', r20)
print('rollmedian', r30)
print('quadratic', r40)
