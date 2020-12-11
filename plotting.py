import numpy as np
import matplotlib.pyplot as plt
from data_fetcher import *

data_stock = api_call_python('GOLD', start_date='2008-01-01', end_date=None, use_cache=True, sampling_rate='weekly')
times_prices = np.array([(row[0], row[4]) for row in data_stock])
times_prices_hsplit = np.array(np.hsplit(times_prices, 2))
dates = np.array([times_prices_hsplit[0][i][0] for i in range(times_prices.shape[0])])
closing_prices = np.array([times_prices_hsplit[1][i][0] for i in range(times_prices.shape[0])])

plt.plot_date(dates, closing_prices, fmt='-b')
plt.show()
