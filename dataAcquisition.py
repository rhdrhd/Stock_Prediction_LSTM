import datetime as dt
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from pytrends.request import TrendReq



def acquire_data_from_yfinance(symbol, start_time, end_time):
    """
    Fetch data from yfinance API

    Args:
        symbol (str): Stock symbol
        start_time (datetime): Start time
        end_time (datetime): End time
    
    Returns:
        pd.DataFrame: Dataframe containing the data
    """

    history = yf.download(symbol, start=start_time, end=end_time, interval='1d',progress=False)
    history_formatted = history.reset_index()
    history_formatted["Date"] = history_formatted["Date"].dt.date
    history_formatted["Date"] = pd.to_datetime(history_formatted["Date"],format='%Y-%m-%d')

    for col in history_formatted.columns:
        if col != 'Date':
            history_formatted.rename(columns={col: f"{symbol}_{col}"}, inplace=True)
    return history_formatted

def acquire_data_from_fred(series, start_time, end_time):
    """
    Fetch data from FRED API
    
    Args:
        series (str): Series name
        start_time (datetime): Start time
        end_time (datetime): End time
    
    Returns:
        pd.DataFrame: Dataframe containing the data
    """
    fred = Fred(api_key='62ba449f2d0cc0dacc3e4ebe7d47d073')
    data = fred.get_series(series, observation_start=start_time, observation_end=end_time)
    data = data.to_frame()
    data_formatted = data.reset_index()
    data_formatted.columns = ['Date', series]
    data_formatted["Date"] = pd.to_datetime(data_formatted["Date"],format='%Y-%m-%d')
    return data_formatted

#not using so far
def acquire_data_from_fmp():
    url = ("https://financialmodelingprep.com/api/v3/income-statement/0000789019?period=annual&apikey=kVVFZ2yreiPZgxX0c0cy9J2K8rR7IAIe")
    r = requests.get(url)
    data = r.json()
    my_dict = {}
    for i in range(len(data)):
        my_dict[i] = data[i]
    data = pd.DataFrame.from_dict(my_dict, orient='index')
    data_formatted = data.rename(columns={'date': 'Date'})
    data_formatted["Date"] = pd.to_datetime(data_formatted["Date"],format='%Y-%m-%d')
    return data_formatted


def acquire_data_from_weather(start_time, end_time):
    """
    Fetch data from Open-Meteo API
    
    Args:
        start_time (datetime): Start time
        end_time (datetime): End time
        
    Returns:
        pd.DataFrame: Dataframe containing the data
    """

    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date={start_time}&end_date={end_time}&daily=temperature_2m_mean,sunshine_duration,wind_speed_10m_max&timezone=America%2FNew_York")
    r = requests.get(url)

    data = r.json()
    weather_data = pd.DataFrame.from_dict(data['daily'])
    weather_data = weather_data.rename(columns={"time":"Date"})
    weather_data["Date"] = pd.to_datetime(weather_data["Date"],format='%Y-%m-%d')
    return weather_data

def acquire_data_from_google_trends(symbol, start_time, end_time):
    """
    Fetch data from Google Trends API
    
    Args:
        symbol (str): Stock symbol
        start_time (datetime): Start time
        end_time (datetime): End time
    
    Returns:
        pd.DataFrame: Dataframe containing the data
    """
    # Only need to run this once, the rest of requests will use the same session.
    pytrend = TrendReq()

    # Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
    pytrend.build_payload(kw_list=[symbol], timeframe=f'{start_time} {end_time}', geo='US')

    # Interest Over Time
    interest_over_time_df = pytrend.interest_over_time()
    interest_over_time_df = interest_over_time_df.reset_index()
    interest_over_time_df = interest_over_time_df.drop(columns={"isPartial"})
    interest_over_time_df.columns = ['Date', symbol]
    interest_over_time_df["Date"] = pd.to_datetime(interest_over_time_df["Date"],format='%Y-%m-%d')


    return interest_over_time_df

def join_dataframes(data):
    """
    Joins the DataFrames in a dictionary into one DataFrame.

    Args:
        data (dict): Dictionary containing the DataFrames.

    Returns:
        pd.DataFrame: Joined DataFrame.
    """
    joined_df = data['MSFT'].copy()

    for key in data.keys():
        if key == 'MSFT': continue
        joined_df = pd.merge(joined_df, data[key], on='Date', how='left')

    print(f"All data joined successfully")    
    return joined_df



def acquire_data_from_api():
    """
    Fetch data from all APIs

    Returns:
        dict: Dictionary containing the data
    """

    start_time = dt.datetime(2019, 4, 1).strftime('%Y-%m-%d')
    end_time = dt.datetime(2023, 3, 31).strftime('%Y-%m-%d')

    MSFT = acquire_data_from_yfinance("MSFT", start_time, end_time)
    nasdaq = acquire_data_from_yfinance("^IXIC", start_time, end_time)
    dowj = acquire_data_from_yfinance("^DJI", start_time, end_time)
    sp500 = acquire_data_from_yfinance("^GSPC", start_time, end_time)
    russell2000 = acquire_data_from_yfinance("^RUT", start_time, end_time)
    crudeoil = acquire_data_from_yfinance("CL=F", start_time, end_time)
    gold = acquire_data_from_yfinance("GC=F", start_time, end_time)
    eurotoUSD = acquire_data_from_yfinance("EURUSD=X", start_time, end_time)
    tenyearbond = acquire_data_from_yfinance("^TNX", start_time, end_time)


    inflation = acquire_data_from_fred("T10YIE",start_time, end_time)
    uncertainty_index = acquire_data_from_fred("USEPUINDXD",start_time, end_time)
    bank_prime_loan_rate = acquire_data_from_fred("DPRIME",start_time, end_time)

    weather_data = acquire_data_from_weather(start_time, end_time)

    #google_trend_data = acquire_data_from_google_trends("Microsoft", start_time, end_time)
    data = {"MSFT": MSFT, "nasdaq": nasdaq, "dowj": dowj, "sp500":sp500, "russell2000":russell2000, "crudeoil":crudeoil, "gold":gold, "eurotoUSD":eurotoUSD, "tenyearbond":tenyearbond, "inflation": inflation, 
            "uncertainty_index": uncertainty_index, "bank_prime_loan_rate": bank_prime_loan_rate, "weather_data": weather_data}
    return_data = {}
    return_data["combined_df"] = join_dataframes(data)
    
    print("Data fetched successfully")

    return return_data
    




