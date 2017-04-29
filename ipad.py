# -*- coding: utf-8 -*-
"""
Forecasting iPad Sales Using Prophet
Nathan Brixius, April 2017
"""
from fbprophet import Prophet
import pandas as pd
import numpy as np

def make_data(growth):
  df = pd.read_csv('ipad_sales.csv')
  if growth=='logistic':
    df['y'] = np.log(df['y'])
  return df


def make_holidays():
  h = pd.DataFrame({
    'holiday': 'release',
    'ds': pd.read_csv(r'ipad_release.csv')['ds'],
    'lower_window': 0.0,
    'upper_window': 366/4
  })
  return h

def ipad(growth='logistic'):
  df = make_data(growth)
  h = make_holidays()
  m = Prophet(growth=growth, holidays=h)
  cap = 1.1 * np.max(df['y'])
  if growth=='logistic':
    df['cap'] = cap
  fit = m.fit(df)
  future = m.make_future_dataframe(periods=4, freq='BQ')
  future['cap'] = cap
  forecast = fit.predict(future)
  m.plot(forecast)
  return m