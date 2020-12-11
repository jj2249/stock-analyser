import numpy as np
import os
import pandas as pd
import quandl
from quandl.errors.quandl_error import NotFoundError 
import sys


def read_csv(file_path, specifyCols=None):
    """
    reads a csv file into a Pandas dataframe. default argument allows for specific columns to be extraced
    """
    data  = pd.read_csv(file_path, usecols=specifyCols)
    return data


def prepend_database_stock(stock_key, database):
    """
    takes in a single stock ticker as a string and prepends the database name
    """
    return database + '/' + stock_key


def retrieve_stock_data(stock_key, start_d=None, end_d=None, sampling_rate=None, use_cache=False):
    """
    uses the quandl API to fetch online stock data, option to use cache to avoid repeated requests to the quandl
    server. Note: Can only use cache when considering one stock as it will overwrite file each time
    """
    print('Loading data...')

    # authenticate requests using API key
    quandl.ApiConfig.api_key = "MLEChCzx-xjxjkefxwpi"

    # check that cache exists by trying to make it
    dir = 'cache'
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

    # add cache file to cache directory
    cachef = os.path.join(dir, 'stock_data.npy')

    # check for cache file and if it is not found then fetch online data and dump to cache
    if use_cache:
        try:
            data = np.load(cachef, 'r')
            print('Cache used successfully')
        except FileNotFoundError:
            data = quandl.get(stock_key, returns='numpy', start_date=start_d, end_date=end_d, collapse=sampling_rate)
            np.save(cachef, data)
            print('Cache used unsuccessfully')


    # fetch online data and ignore cache
    else:
        data = quandl.get(stock_key, returns='numpy', start_date=start_d, end_date=end_d, collapse=sampling_rate)

    return data


def api_call_csv(csv_data, final_csv):
    """
    main function that takes in a pre-formatted csv and handles api calling, then writes collected data out to a new csv
    """
    # define global dataframe for final write-out
    full_data = pd.DataFrame()

    # parse input csv and zip data into 3-tuples
    stock_names = read_csv(csv_data, specifyCols=['Ticker', 'Start Date', 'End Date'])
    stock_tuples = list(stock_names.itertuples(index=False, name=None))

    valid_count = 0

    # retrieve price data for each stock and append it to the global data frame
    for stock in stock_tuples:
        try:
            temp_data = retrieve_stock_data(prepend_database_stock(str(stock[0]), 'LBMA'), stock[1], stock[2], sampling_rate='monthly')
            temp = np.array([item[4] for item in temp_data])
            temp_df = pd.DataFrame({str(stock[0]) : temp})

            full_data = pd.concat([full_data, temp_df], axis=1, sort=False)
            # if the call was valid, increment
            valid_count += 1

        except NotFoundError:
            # pass if no entry is found in the database
            pass

    # write the global dataframe to a file and return the valid count
    full_data.to_csv(final_csv)
    return valid_count

def api_call_python(stock_key, start_date=None, end_date=None, sampling_rate=None, use_cache=False):
	return retrieve_stock_data(prepend_database_stock(stock_key, 'LBMA'), start_d=start_date, end_d=end_date, sampling_rate=sampling_rate, use_cache=use_cache)
