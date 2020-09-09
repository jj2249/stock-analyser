import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import quandl
import re
import sys

"""
def save_as_csv(infile, outfile):
    # open an xlsx file and save a copy as a csv. Handles all file closing internally

    # check file extensions
    if (re.search('_(\d+)\.xlsx$', infile) and re.search('_(\d+)\.csv$', outfile)):
        df = pd.read_excel(infile)
        df.to_csv(outfile) #default separator is a comma

    else:
        print("invalid file names")
        sys.exit() # okay to exit as no files have been opened
"""


def read_csv(file_path):
    data  = pd.read_csv(file_path, usecols=[0, 1, 2])
    return data


def prepend_database_stock(stock_key, database):
    """
    takes in a single stock ticker as a string and prepends the database name
    """
    return database + '/' + stock_key


def prepend_database_multiple_stocks(stock_keys, database):
    """
    takes in a list of stock tickers and prepends the database to each. database must be in string format
    """
    for index, ticker in enumerate(stock_keys):
        stock_keys[index] = database + '/' + ticker
    return stock_keys


def retrieve_stock_data(stock_key, start_d=None, end_d=None, sampling_rate=None, use_cache=False):
    """
    uses the quandl API to fetch online stock data, option to use cache to avoid repeated requests to the quandl
    server. Note: Can only use cache when considering one stock as it will overwrite file each time
    """
    print('Loading data...')
    # authenticate requests using API key
    quandl.ApiConfig.api_key = 'MLEChCzx-xjxjkefxwpi'

    dir = 'cache'
    # check that cache exists by trying to make it
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

    # add file to existing directory
    cachef = os.path.join(dir, 'stock_data.npy')

    # check for cache file and if it is not found then fetch online data and dump to cache
    if use_cache:
        try:
            data = np.load(cachef, 'r')
        except FileNotFoundError:
            data = quandl.get(stock_key, returns='numpy', start_date=start_d, end_date=end_d, collapse=sampling_rate)
            np.save(cachef, data)

    # fetch online data and ignore cache
    else:
        data = quandl.get(stock_key, returns='numpy', start_date=start_d, end_date=end_d, collapse=sampling_rate)
        #np.save(cachef, data)

    # data is in tuple pairs stored in numpy array so separate
    #t = np.array([item[0] for item in data])
    #s = np.array([item[1] for item in data])

    return data


def api_call(csv_data):
    stock_names = read_csv(csv_data)
    stock_tuples = list(stock_names.itertuples(index=False, name=None))
    for stock in stock_tuples:
        print(retrieve_stock_data(prepend_database_stock(stock[0], "WIKI"), stock[1], stock[2], sampling_rate='daily'))


file = "./fund_data.csv"

print(retrieve_stock_data("WIKI/AAPL", "2016-01-01", "2016-01-08"))
