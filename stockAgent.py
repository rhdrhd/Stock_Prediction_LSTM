from dataStorage import *
from dataPreprocessing import *
from dataForecasting import *
from datetime import datetime, timedelta

# Set random seeds for reproducibility
seed_value = 42

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def get_current_price(date, stock_symbol):
    stock_data = yf.download(stock_symbol, start=date, end=date + timedelta(days=1), interval='1d', progress=False)
    current_price = stock_data["Open"]
    return current_price.values

def get_true_values_end_of_day(date, stock_symbol):
    stock_data = yf.download(stock_symbol, start=date, end=date + timedelta(days=1), interval='1d', progress=False)
    true_close = stock_data["Close"]
    true_close_adj = stock_data["Adj Close"]
    return np.array([true_close.values,true_close_adj.values])


def apply_trading_agent(model, date, sliding_window, current_budget, current_stock_count, scaler_data):


    # Get the model's prediction for the next day
    output = model.predict(sliding_window, verbose = None)
    predicted_price = scaler_data["MSFT_Close"].inverse_transform(output[0][0].reshape(-1,1))
    predicted_price = predicted_price.reshape(1)
    # Define decision thresholds
    buy_threshold = 1.01  # e.g., 1% higher than current price
    sell_threshold = 0.99  # e.g., 1% lower than current price

    #replace with current price from api if possible
    current_price = get_current_price(date, "MSFT")

    # Decision-making
    if predicted_price >= current_price * buy_threshold:
        #use 70% of budget to buy the stock
        stocks_to_buy = int(0.7*current_budget / current_price)
        action = stocks_to_buy

    elif predicted_price <= current_price * sell_threshold:
        #sell 70% of current stock if lower than sell_threshold
        action = -int(0.7*current_stock_count)
    else:
        # Hold
        action = 0

    return action, predicted_price

def simulate_decision_making(o_data, scaler_data):
    #the function supposes you buy at the open on everyday
    #but it can be adjusted to real time by modifying get_current_price
    wnd = 7
    feature_num = 2
    pretrained_weights = True
    model = None
    data_selected = select_feature(o_data, feature_num)
    x_train, y_train, x_test, y_test = load_data(data_selected, wnd)
    model = load_LSTM_model(x_train, pretrained_weights=True)

    # Extract the last 'n' days of data
    last_n_days = -7
    sliding_window = data_selected[last_n_days:]
    feature_num = len(data_selected.columns)
    sliding_window = sliding_window.astype('float32')
    sliding_window = np.reshape(sliding_window, (1, sliding_window.shape[0], feature_num))

    symbol = "MSFT"
    budget = 10000
    initial_wealth = budget
    stock_count = 0
    dt = pd.date_range(start='2023-4-1', end='2023-4-30',freq='B')
    holidays = ['2023-04-07'] 
    holidays = pd.to_datetime(holidays)
    dt = dt.drop(holidays)
    total_value= budget
    open('plot_images/stockAgent_result/output.txt', 'w').close()

    for weekdays in dt:
        
        previous_total_value = total_value
        initial_budget = budget
        current_price = get_current_price(weekdays, symbol)
        action, predicted_price= apply_trading_agent(model, weekdays, sliding_window,budget,0, scaler_data)
        budget = budget - action*current_price
        stock_count = stock_count + action

        #your wealth and gain at the end of day
        true_values = get_true_values_end_of_day(weekdays, symbol)
        true_values_norm = true_values.copy()
        total_value = stock_count*true_values[0] + budget
        total_gain = total_value - initial_wealth
        daily_gain = total_value - previous_total_value


        true_values_norm[0] = scaler_data["MSFT_Close"].transform(true_values_norm[0].reshape(-1,1))
        true_values_norm[1] = scaler_data["MSFT_Adj Close"].transform(true_values_norm[1].reshape(-1,1))
        true_values_norm = np.reshape(true_values_norm,(1,1,feature_num))
        sliding_window = np.delete(sliding_window, (0),axis=1)
        sliding_window = np.append(sliding_window, true_values_norm, axis = 1)
        with open('plot_images/stockAgent_result/output.txt', 'a') as file:
            file.write("*********************\n")
            file.write(f"today is {weekdays.date()}\n")
            file.write(f"current price: {current_price}\n")
            file.write(f"predicted price: {predicted_price}\n")
            file.write(f"action is to {action} stock\n")
            file.write(f"true price: {true_values[0]}\n")
            file.write(f"initial budget: {initial_budget}\n")
            file.write(f"final budget: {budget}\n")
            file.write(f"stock count: {stock_count}\n")
            file.write(f"daily gain: {daily_gain}\n")
            file.write(f"total wealth: {total_value}\n")
            file.write(f"total gain: {total_gain}\n")
    print("Simulation of trading agent finished. Please check result in folder plot_images/stockAgent_result")
    
        

        

        
