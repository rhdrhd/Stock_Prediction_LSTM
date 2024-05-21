import random
import datetime as dt
from math import sqrt
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




# Set random seeds for reproducibility
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def load_data(data, window):

    dataset = data.values
    feature_num = len(data.columns)

    #create train dataset
    training_data_len = int(np.ceil( len(dataset) * 0.95 ))
    train_data = dataset[0:training_data_len]
    x_train = []
    y_train = []

    for i in range(window, len(train_data)):
        x_train.append(train_data[i-window:i])
        y_train.append(train_data[i])

    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('float32')
    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature_num))
    y_train = np.reshape(y_train, (y_train.shape[0], feature_num))

    #create test dataset
    test_data = dataset[training_data_len-window: ]
    x_test = []
    y_test = dataset[training_data_len:]
    for i in range(window, len(test_data)):
        x_test.append(test_data[i-window:i])

    # Convert the data to a numpy array
    x_test = np.array(x_test).astype('float32')
    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_num))
    
    return x_train, y_train, x_test, y_test

def load_LSTM_model(x_train, pretrained_weights=False):
    feature_num = x_train.shape[2]
    wnd = x_train.shape[1]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape= (x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128, return_sequences=False))

    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(feature_num))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001,clipnorm=1.0), loss='mean_squared_error')
    model.summary()
    if pretrained_weights:
        model.load_weights(f"pretrained_weights/LSTM_model_weights_{feature_num}features_{wnd}days.h5")

    return model

def train_LSTM_model(x_train, y_train, batch_size_num, epoch_num):
    feature_num = x_train.shape[2]
    wnd = x_train.shape[1]

    model = load_LSTM_model(x_train,pretrained_weights=False)
    # Train the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, validation_split=0.3, batch_size=batch_size_num, epochs=epoch_num, callbacks=[es])
    model.save_weights(f'pretrained_weights/LSTM_model_weights_{feature_num}features_{wnd}days.h5')
    return model

def select_feature(data, feature_num):
    if feature_num == 1:
        feature_names = ["MSFT_Close"]
    elif feature_num == 2:
        feature_names = ["MSFT_Adj Close","MSFT_Close"]
    elif feature_num == 9:
        feature_names = ["MSFT_Close","MSFT_Adj Close","MSFT_Open","MSFT_High","MSFT_Low","^GSPC_Open","^GSPC_Close","^GSPC_High","^GSPC_Low"]
    elif feature_num == 13:
        feature_names = ["MSFT_Close","MSFT_Adj Close","MSFT_Open","MSFT_High","MSFT_Low","^GSPC_Open","^GSPC_Close","^GSPC_High","^GSPC_Low","^IXIC_Open","^IXIC_Close","^IXIC_High","^IXIC_Low"]
    elif feature_num == 22:
        feature_names = ["MSFT_Close","MSFT_Adj Close","MSFT_Open","MSFT_High","MSFT_Low","^GSPC_Open","^GSPC_Close","^GSPC_High","^GSPC_Low","^IXIC_Open","^IXIC_Close","^IXIC_High","^IXIC_Low","^DJI_Open","^DJI_Close","^DJI_High","^DJI_Low", "T10YIE", "^RUT_Open", "^RUT_Close", "^RUT_High", "^RUT_Low"]
    else:
        raise ValueError("feature_num must be 1 or 9")
    
    data = data.set_index('Date')
    return data[feature_names]

def evaluate_performance_by_metrics(y_test, predictions):
    rmse = sqrt(mean_squared_error(y_test, predictions))

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)

    # Calculate R-squared (R^2)
    r_squared = r2_score(y_test, predictions)

    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R^2):", r_squared)

def test_LSTM_model(model, x_test, y_test, feature_names):
    # use direct forecasting on the test dataset
    predictions = model.predict(x_test)
    # Get the root mean squared error (RMSE)
    rmse = sqrt(mean_squared_error(y_test,predictions))
    #evaluate_performance_by_metrics(y_test, predictions)
    #plot the predicted result and the test dataset
    fig = plt.figure(dpi=300)
    for feature in feature_names:
        plt.plot(y_test[:,feature_names.index(feature)], label=feature)
        plt.plot(predictions[:,feature_names.index(feature)], label=feature + ' Prediction')
        break

    plt.figtext(0.5, 0.2, f'avg rmse: {rmse:.4f}', fontsize=9, color='black')
    plt.title('Testset Evaluation with Direct Forecasting')
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.savefig("plot_images/forecasting_result/direct_forecasting_testset.png")
    plt.close()


def predict_next_day(model, dataset, last_n_days, scaler_data):
    # get the last n day stock movements (and auxilary data) 
    # use recursive predicting to predict the next day price
    sliding_window = dataset[last_n_days:]
    feature_num = len(dataset.columns)

    sliding_window = sliding_window.astype('float32')
    sliding_window = np.reshape(sliding_window, (1, sliding_window.shape[0], feature_num))

    #fetch only workdays in the evaluation window
    dt = pd.date_range(start='2023-4-1', end='2023-4-30',freq='B')

    #exclude the holidays, such as here Good Friday in 2023
    holidays = ['2023-04-07'] 
    holidays = pd.to_datetime(holidays)
    dt = dt.drop(holidays)

    april_predict = np.array([])

    for weekdays in dt:
        predictions = model.predict(sliding_window, verbose=None)
        #only take the close predict from all the values
        april_predict= np.append(april_predict,predictions[0][0])
        #print("april predict shape ", april_predict.shape, " ", april_predict)
        predictions = np.reshape(predictions,(1,predictions.shape[0],feature_num))
        sliding_window = np.delete(sliding_window, (0),axis=1)
        sliding_window = np.append(sliding_window, predictions,axis = 1)

    #concatenate DataFrames
    df1 = pd.DataFrame(dt, columns=['Date'])
    df2 = pd.DataFrame(april_predict, columns=['Close_Predict'])
    combined_df = pd.concat([df1, df2], axis=1)
    
    #apply inverse transform to descale the values
    combined_df['Close_Predict'] = scaler_data['MSFT_Close'].inverse_transform(combined_df['Close_Predict'].values.reshape(-1,1))

    return combined_df

def plot_prediction_against_origin(prediction_df):
    start_time = dt.datetime(2023,1,1).date()
    end_time = dt.datetime(2023,4,30).date()
    stock = yf.Ticker("MSFT")
    history = stock.history(start=start_time, end=end_time)
    
    history_formatted = history.reset_index()
    history_formatted["Date"] = pd.to_datetime(history_formatted["Date"].dt.strftime('%Y-%m-%d'))
    combined_data = pd.merge(history_formatted, prediction_df, on='Date', how='left')

    # Calculate RMSE
    filtered_df = combined_data.dropna(subset=['Close', 'Close_Predict'])
    rmse = sqrt(mean_squared_error(filtered_df['Close'], filtered_df['Close_Predict']))
    evaluate_performance_by_metrics(filtered_df['Close'], filtered_df['Close_Predict'])
    # Plotting
    plt.figure(figsize=(12,10))

    # Plot historical data
    plt.plot(combined_data['Date'], combined_data['Close'], label='Historical Close Price')
    # Plot predicted data
    plt.plot(combined_data['Date'], combined_data['Close_Predict'], label='Predicted Close Price', color='red')
    
    plt.figtext(0.7, 0.2, f'avg rmse: {rmse:.4f}', fontsize=9, color='black')
    plt.xticks(rotation=45) 
    plt.title('Stock Price Prediction with Recrusive Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.legend()
    plt.savefig("plot_images/forecasting_result/recursive_forecasting.png")
    plt.close()


def create_joint_plot(predictions):
    start_time = dt.datetime(2023,4,1).date()
    end_time = dt.datetime(2023,4,30).date()
    stock = yf.Ticker("MSFT")
    history = stock.history(start=start_time, end=end_time)
    
    history_formatted = history.reset_index()
    history_formatted["Date"] = pd.to_datetime(history_formatted["Date"].dt.strftime('%Y-%m-%d'))
    plt.figure(figsize=(24,24))
    sns.jointplot(x=history_formatted['Close'], y=predictions, kind='reg')
    plt.suptitle("Joint Plot of Predicted Close Price and Actual Close Price")
    plt.xlabel("Actual Close Price")
    plt.ylabel("Recursively Predicted Close Price")
    plt.tight_layout()
    plt.savefig("plot_images/forecasting_result/joint_plot.png")
    plt.close()

def calculate_plot_residuals(predictions):
    start_time = dt.datetime(2023,4,1).date()
    end_time = dt.datetime(2023,4,30).date()
    stock = yf.Ticker("MSFT")
    history = stock.history(start=start_time, end=end_time)
    
    history_formatted = history.reset_index()
    history_formatted["Date"] = pd.to_datetime(history_formatted["Date"].dt.strftime('%Y-%m-%d'))
    residuals = history_formatted["Close"] - predictions
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    skewness_residual = skew(residuals)

    print("Mean of residuals:", mean_residual)
    print("Median of residuals:", median_residual)
    print("Skewness of residuals:", skewness_residual)

    plt.figure(figsize=(12, 6))

    # Histogram of residuals
    plt.hist(residuals, bins=20, alpha=0.7, color='blue', label='Histogram')

    # Density plot of residuals
    residuals_series = pd.Series(residuals)
    residuals_series.plot(kind='kde', color='black', label='Density Plot')

    # Adding titles and labels
    plt.title('Residuals Distribution')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.axvline(mean_residual, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_residual:.2f}')
    plt.axvline(median_residual, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_residual:.2f}')
    plt.legend()

    # Show plot
    plt.savefig("plot_images/forecasting_result/residual_plot.png")
    plt.close()

def visualise_model_performance(model, data, wnd, scaler_data):
    prediction_df = predict_next_day(model, data, -wnd, scaler_data)
    create_joint_plot(prediction_df['Close_Predict'])
    plot_prediction_against_origin(prediction_df)
    calculate_plot_residuals(prediction_df['Close_Predict'])


