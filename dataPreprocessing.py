import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd



def check_df_missing_values(df):
    # Check for any missing values in each row


    missing_percentage = df.isnull().sum() / len(df)
    for i in range(len(missing_percentage)):
        if missing_percentage[i] > 0:
            print(missing_percentage.index[i], "is missing", 100*missing_percentage[i], "% of its values")
    return missing_percentage.any()
    

def fill_df_missing_values(df):

    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    df = df.fillna(method='bfill')
    print("missing values filled successfully")

    return df


def remove_df_outliers_z_score(df):
    # Calculate Z-scores
    z_scores = stats.zscore(df.select_dtypes(include=[float, int]))
    
    threshold = 3
    outliers = (abs(z_scores) > threshold)
    cleaned_df = df[~outliers.any(axis=1)]
    cleaned_df = cleaned_df.reset_index(drop=True)
    print("outliers removed successfully")
    return cleaned_df

    
def plot_data_comparison(original_df, after_processed_df, column_name="MSFT_Close"):
    # Plot the closing prices in the first subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(original_df['Date'], original_df[column_name], label='before preprocessing', color='blue')
    ax1.legend()
    ax1.set_title(f'{column_name} Before and After Preprocessing')
    ax2.plot(after_processed_df['Date'], after_processed_df[column_name], label='after preprocessing', color='red')
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    
    plt.savefig(f'plot_images/preprocessing_result/{column_name}_before_and_after_preprocessing.png')
    

def check_outliers_df(df, column_name):
    """
    Checks for outliers in a specified column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to check for outliers.
    """
    x = df.index
    y = df[column_name]

    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(y))
    
    # Define a threshold for identifying outliers
    threshold = 3
    outliers_idx = np.where(z_scores > threshold)

    if len(outliers_idx[0]) == 0:
        return False
    else:
        return True

def plot_outliers_df(df, column_name):
    """
    Creates a line plot for a specified column in a DataFrame and marks the outliers.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column_name (str): Name of the column to check for outliers.
    """
    x = df.index
    y = df[column_name]

    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(y))
    
    # Define a threshold for identifying outliers
    threshold = 3
    outliers_idx = np.where(z_scores > threshold)
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Data')
    
    # Plotting the outliers
    ax.scatter(x[outliers_idx], y.iloc[outliers_idx], c="red", alpha=0.5, label='Outliers')
    
    ax.set_xlabel('Index')
    ax.set_ylabel(column_name)
    ax.set_title(f'{column_name} with Outliers')
    ax.legend()

    plt.savefig(f"plot_images/preprocessing_result/plot_{column_name}_outliers_df.png")

def scale_dataframe_separately(df):
    #print(df.shape)
    scalers = {}
    scaled_data = pd.DataFrame(index=df.index)
    #print("index is ", df.index)
    scaled_data['Date'] = df['Date']
    #select numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    #print("numerical columns: ", numerical_columns)

    #scale down each column separately
    for columns in numerical_columns:

        # Create a new scaler for each column
        # customize the range to (0.0001,1) for the seasonality analysis
        scaler = MinMaxScaler(feature_range=(0.0001,1))

        # Fit and transform the data
        scaled_data[columns] = scaler.fit_transform(df[columns].values.reshape(-1, 1)).flatten()

        # Store the scaler
        scalers[columns] = scaler

    return scaled_data, scalers

def preprocess_data(data):
    # fill missing values and remove outliers
    # normalize the data

    joined_data_df = data["combined_df"]
    original_df = joined_data_df.copy()

    if check_df_missing_values(joined_data_df):
        joined_data_df = fill_df_missing_values(joined_data_df)
    
    joined_data_df = remove_df_outliers_z_score(joined_data_df)

    df_after_outliers_removed = joined_data_df.copy()
    plot_data_comparison(original_df, df_after_outliers_removed,"T10YIE")
    plot_data_comparison(original_df, joined_data_df,"MSFT_Close")

    joined_data_df, scaler = scale_dataframe_separately(joined_data_df)
    # remove columns with all 0.0001 (after scaling)
    # because previous customized scaling raise the minimum value zero to 0.0001
    joined_data_df = pd.concat([
        joined_data_df.iloc[:, :1],  # Keep the first column (Date)
        joined_data_df.iloc[:, 1:].loc[:, (joined_data_df.iloc[:, 1:] > 0.0001).any(axis=0)]], axis=1)

    #plot the data comparison before and after preprocessing
  
    return joined_data_df, scaler

