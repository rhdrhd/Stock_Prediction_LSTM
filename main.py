from dataAcquisition import acquire_data_from_api
from dataStorage import empty_database, store_data, test_crud, fetch_collection_from_db
from dataPreprocessing import preprocess_data
from dataForecasting import select_feature, load_data, load_LSTM_model, train_LSTM_model, test_LSTM_model, visualise_model_performance
from dataExploration import explore_data
from stockAgent import simulate_decision_making
def main():
    # acquire related data from APIs
    data = acquire_data_from_api()

    # store data in MongoDB database
    empty_database()
    store_data(data)

    #enable this to test crud api
    #test_crud()

    # fetch data from MongoDB database
    data_from_db = fetch_collection_from_db()

    # preprocess data
    o_data, scaler_data= preprocess_data(data_from_db)

    # perform exploratory data analysis
    # findings are saved in "plot_images/exploration_result"
    explore_data(data_from_db,o_data)

    # forecast data
    #the best model weight is loaded
    wnd = 14
    feature_num = 13
    pretrained_weights = False
    model = None
    data_selected = select_feature(o_data, feature_num)
    x_train, y_train, x_test, y_test = load_data(data_selected, wnd)

    # training mode without pretrained weights
    if pretrained_weights == False:
        model = load_LSTM_model(x_train, pretrained_weights=False)
        model = train_LSTM_model(x_train, y_train, batch_size_num=4, epoch_num=50)
        test_LSTM_model(model, x_test, y_test, data_selected.columns.tolist())
    else:
        model = load_LSTM_model(x_train, pretrained_weights=True)
        test_LSTM_model(model, x_test, y_test, data_selected.columns.tolist())

    visualise_model_performance(model, data_selected, wnd, scaler_data)

    #implement the stock agent
    simulate_decision_making(o_data,scaler_data)

if __name__ == "__main__":
    main()
