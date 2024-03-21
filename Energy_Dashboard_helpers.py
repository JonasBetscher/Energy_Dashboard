import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def load_data():
    # Load data
    #consumption_filename1 = r"IST_Civil_Pav_2017 - IST_Civil_Pav_2017.csv"
    consumption_filename2 = r"IST_Civil_Pav_2018 - IST_Civil_Pav_2018.csv"
    meteo_data_filename = r"IST_meteo_data_2017_2018_2019 - IST_meteo_data_2017_2018_2019.csv"
    #consumption_df1 = pd.read_csv(consumption_filename1)
    consumption_df = pd.read_csv(consumption_filename2)
    #consumption_df = pd.concat([consumption_df1, consumption_df2])
    meteo_data_df = pd.read_csv(meteo_data_filename)

    # Convert date columns to datetime format with corrected format
    consumption_df['Date_start'] = pd.to_datetime(consumption_df['Date_start'], format='%d-%m-%Y %H:%M')
    meteo_data_df['yyyy-mm-dd hh:mm:ss'] = pd.to_datetime(meteo_data_df['yyyy-mm-dd hh:mm:ss'], format='%Y-%m-%d %H:%M:%S')

    # Set the datetime columns as the index
    consumption_df.set_index('Date_start', inplace=True)
    meteo_data_df.set_index('yyyy-mm-dd hh:mm:ss', inplace=True)

    # Resample both datasets to minute frequency
    consumption_df_resampled = consumption_df.resample('min').mean()
    meteo_data_df_resampled = meteo_data_df.resample('min').mean()

    # Merge the resampled datasets
    merged_df = pd.merge(consumption_df_resampled, meteo_data_df_resampled, how='outer', left_index=True, right_index=True)

    # Filter rows to include only data from 2017 and 2018
    merged_df = merged_df.loc['2018-01-01':'2018-12-31']

    # Resample the merged dataset to hourly frequency
    merged_df_hourly = merged_df.resample('h').mean()

    # Filter rows where Power_kW is not equal to 0
    merged_df_hourly = merged_df_hourly[merged_df_hourly['Power_kW'] != 0.0]

    # Define the maximum allowed gap size (in hours)
    max_allowed_gap_hours = 48  # 2 days

    # Calculate the time differences between consecutive timestamps
    time_diff = meteo_data_df.index.to_series().diff()

    # Find the indices where the time difference exceeds the maximum allowed gap size
    large_gaps_indices = time_diff[time_diff > pd.Timedelta(hours=max_allowed_gap_hours)].index

    # Define a list to store the labels of rows to be dropped
    rows_to_drop = []

    # Iterate through the large gaps indices
    for index in large_gaps_indices:
        # Find the corresponding labels in merged_df_hourly within the gap
        labels_within_gap = merged_df_hourly.index[(merged_df_hourly.index >= index - time_diff[index]) &
                                                    (merged_df_hourly.index <= index)]

        # Extend the rows_to_drop list with the labels within the gap
        rows_to_drop.extend(labels_within_gap)

    # Drop the rows corresponding to the identified gaps
    merged_df_hourly = merged_df_hourly.drop(index=rows_to_drop)

    # Create a copy of the original DataFrame to avoid modifying it
    df_interpolate_linear = merged_df_hourly.copy()

    # Replace NaN values using interpolation
    df_interpolate_linear = df_interpolate_linear.interpolate(method='linear')

    # Create a new column 'is_weekend' in the DataFrame and set it to 1 for weekend days (Saturday or Sunday)
    df_interpolate_linear['is_weekend'] = (df_interpolate_linear.index.weekday >= 5).astype(int)

    # Extract features from timestamp
    df_interpolate_linear['Day_of_Week'] = df_interpolate_linear.index.dayofweek
    df_interpolate_linear['Date'] = df_interpolate_linear.index.dayofyear
    df_interpolate_linear['Hour_of_Day'] = df_interpolate_linear.index.hour

    # Perform one-hot encoding for Day of the Week
    #df_interpolate_linear = pd.get_dummies(df_interpolate_linear, columns=['Day_of_Week'], prefix='Day')

    # Perform cyclical encoding for Hour of the Day
    df_interpolate_linear['Hour_Sin'] = np.sin(2 * np.pi * df_interpolate_linear['Hour_of_Day'] / 24)
    df_interpolate_linear['Hour_Cos'] = np.cos(2 * np.pi * df_interpolate_linear['Hour_of_Day'] / 24)

    # Perform cyclical encoding for Day of week
    df_interpolate_linear['Day_of_Week_Sin'] = np.sin(2 * np.pi * df_interpolate_linear['Day_of_Week'] / 7)
    df_interpolate_linear['Day_of_Week_Cos'] = np.cos(2 * np.pi * df_interpolate_linear['Day_of_Week'] / 7)

    # Perform cyclical encoding for Date
    df_interpolate_linear['Date_Sin'] = np.sin(2 * np.pi * df_interpolate_linear['Date'] / 365)
    df_interpolate_linear['Date_Cos'] = np.cos(2 * np.pi * df_interpolate_linear['Date'] / 365)

    # Drop the original timestamp and Hour_of_Day columns
    df_interpolate_linear = df_interpolate_linear.drop(columns=['Hour_of_Day', 'Day_of_Week', 'Date'])

    # Add a column with Power_kW of the previous hour
    df_interpolate_linear['Power_kW_-1'] = df_interpolate_linear['Power_kW'].shift(1)

    # Convert the index to WET timezone
    df_interpolate_linear.index = df_interpolate_linear.index.tz_localize('Europe/Lisbon', ambiguous='NaT', nonexistent='shift_forward')

    # Convert the index to UTC timezone (DST will be handled automatically)
    df_interpolate_linear.index = df_interpolate_linear.index.tz_convert('UTC')

    # Sample list of public holidays and academic holidays
    holiday_list = [
        '2017-01-01', '2017-02-14', '2017-02-28', '2017-03-19', '2017-03-20', '2017-03-26',
        '2017-04-14', '2017-04-16', '2017-04-25', '2017-05-01', '2017-05-07', '2017-05-12',
        '2017-05-12', '2017-06-10', '2017-06-13',
        '2017-06-15', '2017-06-21', '2017-08-15',
        '2017-09-22', '2017-10-05',
        '2017-10-29', '2017-11-01', '2017-12-01', '2017-12-08', '2017-12-21', '2017-12-24',
        '2017-12-25', '2017-12-31',
        '2018-01-01', '2018-02-13', '2018-02-14', '2018-03-19', '2018-03-19', '2018-03-20', '2018-03-25',
        '2018-03-30', '2018-04-01', '2018-04-25', '2018-05-01', '2018-05-06', '2018-05-31', '2018-06-10', '2018-06-13',
        '2018-06-21', '2018-08-15', '2018-09-23', '2018-10-05', '2018-10-28',
        '2018-11-01', '2018-12-01', '2018-12-08', '2018-12-21', '2018-12-24', '2018-12-25',
        '2018-12-31'
    ]

    # Convert the list of holidays to a pandas DateTimeIndex
    holiday_index = pd.to_datetime(holiday_list)

    # Create a new column 'is_holiday' in the DataFrame and set it to 1 for holiday days
    df_interpolate_linear['is_holiday'] = df_interpolate_linear.index.isin(holiday_index).astype(int)

    # Iterate over the holiday dates
    for holiday_date in holiday_index:
        # Set the holiday information for all hours of the holiday date
        df_interpolate_linear.loc[df_interpolate_linear.index.date == holiday_date.date(), 'is_holiday'] = 1

    # Drop NaN value (First entry of Power-1):
    df_interpolate_linear = df_interpolate_linear.dropna()

    return df_interpolate_linear

def calculate_feature_importance_RF(data):
    # Split the data into features (X) and target variable (y)
    X = data.drop(columns=['Power_kW'])
    y = data['Power_kW']

    # Train the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)

    # Get feature importances
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def train_model(data, model_name, features):
    # Split the data into features (X) and target variable (y)
    X = data[[feature for feature in features if feature in data.columns]]

    # Filter the y data to include only columns that are in the features list
    y = data['Power_kW']

    # Train the model
    if model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Support Vector Machine (RBF kernel)':
        model = SVR()
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == 'Multi-layer Perceptron':   
        model = MLPRegressor(random_state=42)
    elif model_name == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError('Invalid model name. Please select a valid model.')

    model.fit(X, y)

    # Define model name based on features and model_name
    model_name = model_name + ' (' + ', '.join(features) + ')'

    return [model_name, model]

def get_predictions(data, model, features):
    # Split the data into features (X) and target variable (y)
    X = data[[feature for feature in features if feature in data.columns]]

    # Get the predictions
    predictions = model.predict(X)

    return predictions

def create_test_data(dataframe):
    consumption_df3 = dataframe.copy()
    # Convert date columns to datetime format with corrected format
    consumption_df3['Date'] = pd.to_datetime(consumption_df3['Date'], format='%Y-%m-%d %H:%M:%S')

    # Set the datetime columns as the index
    consumption_df3.set_index('Date', inplace=True)

    # Create a new column 'is_weekend' in the DataFrame and set it to 1 for weekend days (Saturday or Sunday)
    consumption_df3['is_weekend'] = (consumption_df3.index.weekday >= 5).astype(int)

    # Extract features from timestamp
    consumption_df3['Day_of_Week'] = consumption_df3.index.dayofweek
    consumption_df3['Hour_of_Day'] = consumption_df3.index.hour
    consumption_df3['Date'] = consumption_df3.index.dayofyear

    # Perform cyclical encoding for Hour of the Day
    consumption_df3['Hour_Cos'] = np.cos(2 * np.pi * consumption_df3['Hour_of_Day'] / 24)
    consumption_df3['Hour_Sin'] = np.sin(2 * np.pi * consumption_df3['Hour_of_Day'] / 24)

    # Perform cyclical encoding for Day of week
    consumption_df3['Day_of_Week_Sin'] = np.sin(2 * np.pi * consumption_df3['Day_of_Week'] / 7)
    consumption_df3['Day_of_Week_Cos'] = np.cos(2 * np.pi * consumption_df3['Day_of_Week'] / 7)

    # Perform cyclical encoding for Date
    consumption_df3['Date_Sin'] = np.sin(2 * np.pi * consumption_df3['Date'] / 365)
    consumption_df3['Date_Cos'] = np.cos(2 * np.pi * consumption_df3['Date'] / 365)

    # Drop the original timestamp and Hour_of_Day columns
    consumption_df3 = consumption_df3.drop(columns=['Hour_of_Day', 'Day_of_Week', 'Date'])

    # Sample list of public holidays and academic holidays
    holiday_list_2019 = [
        '2019-01-01', '2019-02-14', '2019-03-05', '2019-03-19', '2019-03-20',
        '2019-04-19', '2019-04-21', '2019-04-25', '2019-05-01', '2019-05-05', '2019-06-10', '2019-06-13',
        '2019-06-20', '2019-08-15', '2019-10-05', '2019-11-01', '2019-12-01', '2019-12-08', '2019-12-24',
        '2019-12-25', '2019-12-31'
    ]

    # Convert the list of holidays to a pandas DateTimeIndex
    holiday_index = pd.to_datetime(holiday_list_2019)

    # Create a new column 'is_holiday' in the DataFrame and set it to 1 for holiday days
    consumption_df3['is_holiday'] = consumption_df3.index.isin(holiday_index).astype(int)

    # Iterate over the holiday dates
    for holiday_date in holiday_index:
        # Set the holiday information for all hours of the holiday date
        consumption_df3.loc[consumption_df3.index.date == holiday_date.date(), 'is_holiday'] = 1

    # Convert the index to WET timezone
    consumption_df3.index = consumption_df3.index.tz_localize('Europe/Lisbon', ambiguous='NaT', nonexistent='shift_forward')

    # Convert the index to UTC timezone (DST will be handled automatically)
    consumption_df3.index = consumption_df3.index.tz_convert('UTC')

    # Add a column with Power_kW of the previous hour
    consumption_df3['Power_kW'] = consumption_df3['Civil (kWh)']
    consumption_df3 = consumption_df3.drop(columns=['Civil (kWh)'])
    consumption_df3['Power_kW_-1'] = consumption_df3['Power_kW'].shift(1)

    consumption_df3 = consumption_df3.drop(columns=['Power_kW'])

    # Drop NaN value (First entry of Power-1):
    consumption_df3 = consumption_df3.dropna()

    return consumption_df3
