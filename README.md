# Data Acquisition and Processing Systems (DAPS) (ELEC0136)

Predicting future stock prices has always been a challenging yet crucial task in the financial sector, which offers valuable insights for investors and companies. This project aims to address the challenges by developing a customized Long Short Term Memory (LSTM) model to predict the stock price of Microsoft Corporation (MSFT).

## Project structure
The current project structure is shown below
```
├── README.md
├── dataAcquisition.py
├── dataStorage.py
├── dataPreprocessing.py
├── dataExploration.py
├── dataForecasting.py
├── stockAgent.py
├── environment.yml
├── requirements.txt
├── pretrained_weights
├── plot_images
│   ├── exploration_result
│   ├── forecasting_result
│   ├── preprocessing_result
└── main.py
```

**main.py**: Contains the core of the project, including data acquisition, data storage, data preprocess, data exploration, model prediction and stock agent. 

## How to start
1. Create a new conda environment from environment.yml file.
```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate daps-final
```
3. Run main.py if all the dependencies required for the current project are already installed. 

```
python main.py
```
## NOTES

The main file is defaulted to train the model. To make sure about 100% reproducibility please load pretrained weights by changing the wnd and feature_num value in main.py. For exmaple, to load model with 13 features and 14 days window, simply set feature_num = 13, and wnd = 14.

By default, the model with 13 features and 14 days window is set to be trained. 
