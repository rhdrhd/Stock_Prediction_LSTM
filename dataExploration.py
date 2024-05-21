from dataPreprocessing import *
from dataStorage import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from calendar import day_abbr
import datetime as dt
import calendar
import matplotlib.dates as mdates



def plot_trend_and_seasonality(df, column):
    df = df.copy()
    df = df.set_index('Date')
    result = seasonal_decompose(df[column], model='multiplicative', period=66)
    figure = plt.figure(figsize=(40,20))
    result.plot()
    
    plt.savefig(f'plot_images/exploration_result/{column}_trend_and_seasonality_analysis.png')
    plt.close()
    #result.seasonal["2019-12-01":"2020-06-01"].plot()

def plot_2022_seasonality(df, column):

    start_date = '2022-01-01'
    end_date = '2023-01-01'
    mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    filtered_df = df.loc[mask]

    filtered_df = filtered_df.set_index('Date')

    # Apply seasonal decomposition
    result = seasonal_decompose(filtered_df[column], model='multiplicative', period=66)

    plt.figure(figsize=(12, 6))
    plt.plot(result.seasonal)
    plt.title(f'Seasonality of {column} during 2022')

    # Set major ticks format
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xlabel('Date')
    plt.ylabel('Seasonality')
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability

    # Make sure the figure uses the specified formatter and locator
    plt.gcf().autofmt_xdate()
    # Save the plot
    plt.savefig(f'plot_images/exploration_result/{column}_2022_seasonality.png')
    plt.close()

def plot_correlation_heatmap(dataframe):
    # Assuming 'data' is your DataFrame
    #correlation_matrix = df.corr()
    df = dataframe.copy()
    df = df.set_index('Date')
    correlation_matrix= abs(df.corr()).sort_values('MSFT_Close', ascending=False)
    plt.figure(figsize=(40, 40))
    
    ax = sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 8}, cmap='coolwarm', linewidth=.5, fmt='.2f')
    ax.xaxis.tick_top()  # Move x-ticks to the top
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top

    # Optionally, adjust rotation of x labels if they overlap
    plt.xticks(rotation=90)

    plt.savefig('plot_images/exploration_result/stock_features_correlation_heatmap.png')
    plt.close()

def plot_dependency_on_year_month(df):
    year_month_data = df.copy()
    year_month_data['Date'] = pd.to_datetime(year_month_data['Date'])
    year_month_data['year'] = year_month_data['Date'].dt.year
    year_month_data['month'] = year_month_data['Date'].dt.month

    # Group by year and month and aggregate the variable of interest
    # Replace 'closing_price' with your variable of interest
    monthly_data = year_month_data.groupby(['year', 'month'])['MSFT_Close'].mean().unstack()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap
    sns.heatmap(monthly_data, annot=True, cmap=plt.cm.viridis, fmt='.2f', linewidths=.5)

    # Add title and labels as needed
    ax.set_title('Dependency of Closing Price on Year and Month')
    cbax = fig.axes[1]
    cbax.set_ylabel('Closing Price')

    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_yticklabels(np.arange(2019, 2024, 1), rotation=0)
    # Show the plot
    fig.savefig('plot_images/exploration_result/dependency_on_year_month_heatmap.png')
    plt.close()

def plot_dependency_on_month_week(df):
    month_week_data = df.copy()
    month_week_data['Date'] = pd.to_datetime(month_week_data['Date'])
    month_week_data['month'] = month_week_data['Date'].dt.month
    month_week_data['day_of_week'] = month_week_data['Date'].dt.dayofweek

    # Group by year and month and aggregate the variable of interest
    # Replace 'closing_price' with your variable of interest
    weekly_data = month_week_data.groupby(['day_of_week', 'month'])['MSFT_Close'].mean().unstack()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap
    sns.heatmap(weekly_data, annot=True, cmap=plt.cm.viridis, fmt='.2f', linewidths=.5)

    # Add title and labels as needed
    ax.set_title('Dependency of Closing Price on Month and Week')
    cbax = fig.axes[1]
    cbax.set_ylabel('Closing Price')

    ax.set_xlabel('Month')
    ax.set_ylabel('Week')
    ax.set_yticklabels(day_abbr[0:5], rotation=0)
    # Show the plot
    fig.savefig('plot_images/exploration_result/dependency_on_month_week_heatmap.png')
    plt.close()

def test_stationary(dataframe):
    df = dataframe.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # 1. Visual Inspection
    axs[0].plot(df['MSFT_Close'], label='Close Price')
    axs[0].set_title('Stock Close Price Over Time')
    axs[0].legend()

    # 2. Applying Differencing
    df['Close_diff'] = df['MSFT_Close'] - df['MSFT_Close'].shift(1)
    df['Close_diff'].dropna(inplace=True)

    # Re-plotting after differencing
    axs[1].plot(df['Close_diff'], label='Differenced Close Price')
    axs[1].set_title('Differenced Stock Close Price Over Time')
    axs[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.savefig('plot_images/exploration_result/stationary.png')
    plt.close()

    # 3. Testing for Stationarity with ADF Test
    adf_test = adfuller(df['Close_diff'].dropna())
    print('ADF Statistic:', adf_test[0])
    print('p-value:', adf_test[1])


def plot_obv_and_close(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['MSFT_Close'][i] > data['MSFT_Close'][i-1]:
            obv.append(obv[-1] + data['MSFT_Volume'][i])
        elif data['MSFT_Close'][i] < data['MSFT_Close'][i-1]:
            obv.append(obv[-1] - data['MSFT_Volume'][i])
        else:
            obv.append(obv[-1])
    data_plot = np.array(obv)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the first set of data with ax1 and set its color
    ax1.plot(data['Date'], data['MSFT_Close'], color='blue',  label='MSFT_Close')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('MSFT_Close', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a second Y-axis that shares the same X-axis
    ax2 = ax1.twinx()
    ax2.plot(data['Date'], data_plot, color='red',  label='On-Balance Volume')
    ax2.set_ylabel('On-Balance Volume', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # Title for the entire plot
    plt.title('MSFT Close Price and On-Balance Volume')

    # Show the plot
    plt.savefig('plot_images/exploration_result/MSFT_obv_and_close.png')
    plt.close()

def plot_accumulation_distribution_line(df):
    # Calculate the Money Flow Multiplier
    mfm = ((df['MSFT_Close'] - df['MSFT_Low']) - (df['MSFT_High'] - df['MSFT_Close'])) / (df['MSFT_High'] - df['MSFT_Low'])

    # Calculate the Money Flow Volume
    mfv = mfm * df['MSFT_Volume']

    # Calculate the Accumulation/Distribution Line (A/D)
    ad = mfv.cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # 2 Rows, 1 Column

    # Plot the closing prices in the first subplot
    ax1.plot(df['Date'], df['MSFT_Close'], color='blue', label='Closing Price')
    ax1.set_ylabel('Closing Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.set_title('Stock Closing Prices')

    # Plot the Accumulation/Distribution Line in the second subplot
    ax2.plot(df['Date'], ad, color='green', label='A/D Line')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('A/D', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper left')
    ax2.set_title('Accumulation/Distribution Line')

    plt.savefig('plot_images/exploration_result/MSFT_accumulation_distribution_line.png',dpi=200)
    plt.close()

def plot_MACD(df):
    # Calculate the MACD and Signal Line
    # Short term exponential moving average (EMA)
    ShortEMA = df.MSFT_Close.ewm(span=12, adjust=False).mean()
    # Long term exponential moving average (EMA)
    LongEMA = df.MSFT_Close.ewm(span=26, adjust=False).mean()
    # Calculate the MACD line
    MACD = ShortEMA - LongEMA
    # Calculate the signal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    histogram = MACD - signal

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the closing prices in the first subplot
    ax1.plot(df['Date'], df['MSFT_Close'], label='Closing Price', color='blue')
    ax1.set_title('Stock Closing Prices')
    ax1.set_ylabel('Closing Price')
    ax1.legend()

    colors = ['green' if val > 0 else 'red' for val in histogram]

    # Plot the MACD in the second subplot
    ax2.plot(df['Date'], MACD, label='MACD', color='green')
    ax2.plot(df['Date'], signal, label='Signal Line', color='red')
    ax2.bar(df['Date'], histogram, label='Histogram', color=colors)
    ax2.axhline(0, color='blue', linewidth=0.8)
    ax2.set_title('MACD and Signal Line')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('MACD Value')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    plt.savefig('plot_images/exploration_result/MSFT_MACD.png',dpi=200)
    plt.close()

def plot_Aroon(df):

    # Calculate Aroon Indicator
    # Typically a 25-day period is used
    n = 25  
    aroon_up = []
    aroon_down = []

    for i in range(len(df)):
        start = max(0, i - n + 1)
        end = i + 1
        period_high = df['MSFT_Close'][start:end].max()
        period_low = df['MSFT_Close'][start:end].min()
        high_index = df['MSFT_Close'][start:end].idxmax()
        low_index = df['MSFT_Close'][start:end].idxmin()

        aroon_up_value = 100 * ((n - (i - high_index)) / n)
        aroon_down_value = 100 * ((n - (i - low_index)) / n)

        aroon_up.append(aroon_up_value)
        aroon_down.append(aroon_down_value)

    df['Aroon_Up'] = np.array(aroon_up)
    df['Aroon_Down'] = np.array(aroon_down)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(df['Date'],df['MSFT_Close'], label='Closing Price', color='blue')
    ax1.set_title('Stock Closing Prices')
    ax1.set_ylabel('Closing Price')
    ax1.legend()

    ax2.plot(df['Date'],df['Aroon_Up'], label='Aroon Up', color='green')
    ax2.plot(df['Date'],df['Aroon_Down'], label='Aroon Down', color='red')

    ax2.axhline(50, color='black', linewidth=0.8, linestyle='--')

    ax2.set_title('Aroon Indicator')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Aroon Value')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.savefig('plot_images/exploration_result/MSFT_Aroon.png',dpi=200)
    plt.close()

def plot_adx(data, n=14):
    df = data.copy()
    df['+DM'] = df['MSFT_High'].diff()
    df['-DM'] = df['MSFT_Low'].diff()
    df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0.0)
    df['-DM'] = np.abs(df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0.0))

    tr1 = df['MSFT_High'] - df['MSFT_Low']
    tr2 = np.abs(df['MSFT_High'] - df['MSFT_Close'].shift())
    tr3 = np.abs(df['MSFT_Low'] - df['MSFT_Close'].shift())
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    atr = tr.rolling(n).mean()

    df['+DI'] = 100 * (df['+DM'].rolling(n).mean() / atr)
    df['-DI'] = 100 * (df['-DM'].rolling(n).mean() / atr)
    df['DX'] = 100 * np.abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    adx = df['DX'].rolling(n).mean()

    adx, di_plus, di_minus = adx, df['+DI'], df['-DI']

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the closing prices in the first subplot
    ax1.plot(df['Date'],df['MSFT_Close'], label='Closing Price', color='blue')
    ax1.set_title('Stock Closing Prices')
    ax1.set_ylabel('Closing Price')
    ax1.legend()

    # Plot the ADX and DI lines in the second subplot
    ax2.plot(df['Date'],adx, label='ADX', color='green')
    ax2.plot(df['Date'],di_plus, label='+DI', color='blue')
    ax2.plot(df['Date'],di_minus, label='-DI', color='red')

    ax2.set_title('Average Directional Index (ADX) and DI Lines')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ADX / DI Values')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plot_images/exploration_result/MSFT_ADX.png')
    plt.close()

def get_holidays():
    start_date = dt.datetime(2019, 4, 1)
    end_date = dt.datetime(2023, 3, 31)
    holidays = {}

    current_date = start_date
    while current_date <= end_date:
        if current_date.month == 7 and current_date.day == 4:
            holidays[current_date] = 'Independence Day'
        elif current_date.month == 12 and current_date.day == 25:
            holidays[current_date] = 'Christmas Day'
        current_date += dt.timedelta(days=1)

    return holidays

def plot_nearest_holiday(df):
    holidays = get_holidays()
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df['Date'], df['MSFT_Close'], label='MSFT Close Price')
    for holiday_date, holiday_name in holidays.items():

        previous_business_days = df[df['Date'] < holiday_date]

        if not previous_business_days.empty:
            # Find the nearest previous business day that is in the DataFrame
            previous_business_day = previous_business_days.iloc[-1]['Date']
            closing_price = previous_business_days.iloc[-1]['MSFT_Close']
            # Annotate the previous business day
            ax.axvline(x=previous_business_day, color='r', linestyle='--', lw=1)
            ax.text(previous_business_day, 
                    closing_price, 
                    holiday_name,
                    rotation=90,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    fontsize=10,
                    color='red')
    ax.set_title('MSFT Close Price with Christmas and Independence Day Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('MSFT Close Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()  
    plt.savefig('plot_images/exploration_result/MSFT_Close_with_Holidays.png')
    plt.close()

def plot_monthly_return(df):
    monthly_returns = df.copy()
    monthly_returns.set_index('Date', inplace=True)
    monthly_returns = monthly_returns['MSFT_Close'].resample('M').last().pct_change()
    monthly_returns = monthly_returns.reset_index()
    monthly_returns['Date']= pd.to_datetime(monthly_returns['Date'])
    monthly_returns['year'] = monthly_returns['Date'].dt.year
    monthly_returns['month'] = monthly_returns['Date'].dt.month

    # Replace 'closing_price' with your variable of interest
    monthly_returns = monthly_returns.groupby(['year', 'month'])['MSFT_Close'].mean().unstack()
    # Reshape the DataFrame for the heatmap
    #heatmap_data = grouped.pivot('Year', 'Month', return_column)


        # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap
    sns.heatmap(monthly_returns, annot=True, cmap="coolwarm", fmt='.2f', linewidths=.5)

    # Add title and labels as needed
    ax.set_title("Monthly Return Heatmap")
    cbax = fig.axes[1]
    cbax.set_ylabel('Closing Price')

    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    ax.set_yticklabels(np.arange(2019, 2024, 1), rotation=0)
    # Show the plot
    fig.savefig('plot_images/exploration_result/monthly_return.png')
    plt.close()


def explore_data(original_data, preprocessed_df):
    original_df = original_data["combined_df"]
    plot_trend_and_seasonality(preprocessed_df, 'MSFT_Close')
    plot_2022_seasonality(preprocessed_df, 'MSFT_Close')
    plot_correlation_heatmap(original_df)
    plot_dependency_on_year_month(original_df)
    plot_dependency_on_month_week(original_df)
    plot_monthly_return(original_df)
    test_stationary(original_df)
    plot_obv_and_close(original_df)
    plot_accumulation_distribution_line(original_df)
    plot_MACD(original_df)
    plot_Aroon(original_df)
    plot_adx(original_df)
    plot_nearest_holiday(original_df)

