import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
import csv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import chi2
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, LSTM
from kerastuner.tuners import Hyperband


positive_variables = [
    'mood',
    'activity',
    'screen',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unknown',
    'appCat.utilities',
    'appCat.weather'
]
all_variables = [
    'mood',
    'circumplex.arousal',
    'circumplex.valence',
    'activity',
    'screen',
    'appCat.builtin',
    'appCat.communication',
    'appCat.entertainment',
    'appCat.finance',
    'appCat.game',
    'appCat.office',
    'appCat.other',
    'appCat.social',
    'appCat.travel',
    'appCat.unkown',
    'appCat.utilities',
    'appCat.weather',
    'call',
    'sms'
]

filename = "dataset_mood_smartphone.csv"
filename_cleaned = "dataset_mood_smartphone_cleaned.csv"
filename_cleaned_imputed = "dataset_mood_smartphone_cleaned_imputed.csv"
filename_aggregate = "daily_aggregate.csv"
filename_aggregate_avg = "daily_aggregate_avg.csv"
df = pd.read_csv(filename)

def count_id(df):
    """Count the number of unique IDs in the dataset"""
    id_count = df['id'].nunique()
    return id_count

def get_id(df):
    """Get the list of unique IDs in the dataset"""
    id_list = df['id'].unique()
    return id_list

def get_variables(df, variables):
    """Get the list of variables in the dataset"""
    variable_list = df['variable'].isin(variables)
    return variable_list

def timestep_per_id_count(df, variables):
    """Count the number of timesteps per ID in the dataset"""
    id_list = get_id(df)
    timestep_count = []
    for id in id_list:
        for variable in variables:
            timestep_count.append(len(df[(df['id'] == id) & (df['variable'] == variable)]))
    return timestep_count

def plot_scatter_timesteps_per_id(df, variables):
    """Plot a scatter plot of the number of timesteps per ID in the dataset"""
    id_list = get_id(df)
    variable_list = get_variables(df, variables)
    timestep_count = timestep_per_id_count(df, variables)
    plt.scatter(id_list, timestep_count)
    plt.xlabel('ID')
    plt.ylabel('Timestep Count')
    plt.title('Timestep Count per ID')
    plt.savefig('timestep_count_per_id.png')

def aggragete_boolenas_per_day_per_id(df, variables):
    """Aggregate the boolean variables per day per ID in the dataset"""
    id_list = get_id(df)
    booleans = ["sms", "call"]
    for id in id_list:
        for variable in booleans:
            df[(df['id'] == id) & (df['variable'] == variable)].groupby('time').agg('sum')


def count_nan_values(df):
    """Count the number of NaN values in the dataset"""
    nan_values = df.isna().sum()
    print(nan_values)
    return nan_values

def get_earliest_screen_timeentry_perday():
    """Get the earliest screen time entry per day per ID in the dataset"""
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['time'] = df['time'].dt.time
    df_screen = df[df['variable'] == 'screen']
    df_screen = df_screen.groupby(['id', 'date']).agg({'time': 'min'}).reset_index()
    return df_screen

def plot_mood_over_time():
    """Plot the mood over time in the dataset"""
    df_mood = df[df['variable'] == 'mood']
    df_mood['time'] = pd.to_datetime(df_mood['time'])
    df_mood = df_mood.groupby(['time']).agg({'value': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    plt.plot(df_mood['time'], df_mood['value'])
    plt.xlabel('Time')
    plt.ylabel('Mood')
    plt.title('Mood over Time')
    plt.savefig('mood_over_time.png')

def plot_mood_over_time_comparison(attribute):
    """Plot the mood over time in the dataset, comparing minute-level data to daily averages."""
    df_attribute = df[df['variable'] == attribute]
    df_attribute['time'] = pd.to_datetime(df_attribute['time'])

    df_minute_avg = df_attribute.groupby(['time']).agg({'value': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    plt.plot(df_minute_avg['time'], df_minute_avg['value'], label=f'{attribute} over Time')

    df_attribute['date'] = df_attribute['time'].dt.date
    df_daily_avg = df_attribute.groupby(['date']).agg({'value': 'mean'}).reset_index()
    df_daily_avg['date'] = pd.to_datetime(df_daily_avg['date'])

    plt.plot(df_daily_avg['date'], df_daily_avg['value'], 'r', label='Daily Average Mood')

    plt.ylabel(f'{attribute}')
    plt.title(f'{attribute} over Time and Daily Average Mood')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mood_over_time_comparison.png')

def plot_attribute_over_time_comparison(attribute):
    """Plot the specified attribute over time in the dataset, comparing minute-level data to daily minimums."""
    df = pd.read_csv('dataset_mood_smartphone_cleaned.csv')
    df_attribute = df[df['variable'] == attribute]
    df_attribute['time'] = pd.to_datetime(df_attribute['time'])

    booleans = ["sms", "call"]

    if attribute in booleans:
        df_minute_min = df_attribute.groupby(['time']).agg({'value': 'sum'}).reset_index()
    else:
        df_minute_min = df_attribute.groupby(['time']).agg({'value': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    plt.plot(df_minute_min['time'], df_minute_min['value'], label=f'{attribute} over Time')

    if attribute in booleans:
        df_attribute['date'] = df_attribute['time'].dt.date
        df_daily_min = df_attribute.groupby(['date']).agg({'value': 'sum'}).reset_index()
    else:
        df_attribute['date'] = df_attribute['time'].dt.date
        df_daily_min = df_attribute.groupby(['date']).agg({'value': 'mean'}).reset_index()
    df_daily_min['date'] = pd.to_datetime(df_daily_min['date'])

    plt.plot(df_daily_min['date'], df_daily_min['value'], 'r', label='Daily Minimum')

    plt.xlabel('Time')
    plt.ylabel(f'{attribute}')
    plt.title(f'{attribute} over Time and Daily Minimum')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{attribute}_over_time_comparison.png')

    plt.show()

def replace_negative_vals():
    """Replace negative values of positive variables with NaN"""
    df = pd.read_csv('dataset_mood_smartphone.csv')
    for variable in positive_variables:
        df.loc[(df['variable'] == variable) & (df['value'] < 0), 'value'] = np.nan
        df_positive = df
    df_positive.to_csv('dataset_mood_smartphone_cleaned.csv', index=False)
    return df_positive


def plot_dailyavg_variable_distribution(variable):
    """Plot the daily average mood distribution in the dataset"""
    df = pd.read_csv('daily_aggregate.csv')
    df_variable = df[variable]
    plt.figure(figsize=(10, 8))
    sns.distplot(df_variable, bins=20, kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
    plt.xlabel(variable)
    plt.title(f'{variable} Distribution')
    plt.savefig(f'{variable}_distribution.png')


def make_daily_aggregate_csv():
    df = pd.read_csv('dataset_mood_smartphone_cleaned.csv')
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = pd.to_datetime(df['time'].dt.date)

    df_boolean = df[df['variable'].isin(['call', 'sms'])]
    df_real = df[~df['variable'].isin(['call', 'sms'])]

    df_boolean = df_boolean.groupby(['id', 'date', 'variable']).agg({'value': 'sum'}).reset_index()
    df_real = df_real.groupby(['id', 'date', 'variable']).agg({'value': 'mean'}).reset_index()

    df_aggregate = pd.concat([df_boolean, df_real])
   
    min_date = df['date'].min()
    max_date = df['date'].max()

    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    all_ids = df['id'].unique()

    all_date_ids = pd.MultiIndex.from_product([all_ids, all_dates], names=['id', 'date'])
    all_date_ids = pd.DataFrame(index=all_date_ids).reset_index()

    df_aggregate = pd.merge(all_date_ids, df_aggregate, on=['id', 'date'], how='outer')

    df_aggregate = df_aggregate.pivot_table(index=['id', 'date'], columns='variable', values='value', aggfunc='first', dropna=False).reset_index()

    df_aggregate.to_csv('daily_aggregate.csv', index=False)
    return df_aggregate


def sorted_data():
    """ Does the same as make daily aggregate but without averaging"""

    df = pd.read_csv('dataset_mood_smartphone.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['id', 'time'], ascending = True)

    df = df.pivot_table(index=['id', 'time'],
                          columns='variable',
                          values='value',
                          aggfunc='first').reset_index()
    df.columns.name = None
    df = df.reset_index()
    df.drop('index', axis = 1, inplace = True)
    df.to_csv('sorted_data.csv', index=False)

def average_through_ids():
    """Averge the daily values of variables through IDs"""
    df = pd.read_csv('.csv')
    df = df.groupby('date').agg('mean').reset_index()
    df.to_csv('_avg.csv', index=False)
    
def standardise_data():
    """Normalise the data"""
    df = pd.read_csv('daily_aggregate_imputed.csv')

    # Exclude 'id' and 'date' columns
    df = df.drop(columns=['id', 'date'])

    # Replace empty strings with NaN
    df.replace("", np.nan, inplace=True)

    # Calculate mean and standard deviation for each column excluding NaN values
    mean = df.mean(skipna=True)
    std = df.std(skipna=True)

    # Normalize the existing values
    df = (df - mean) / std

    # Add back the 'date' column
    df = pd.concat([df], axis=1)

    # Save the normalized dataset
    df.to_csv('daily_aggregate_normalised.csv', index=False)

def normalise_data(df):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
def remove_outliers_forest():
    df = pd.read_csv('daily_aggregate_imputed.csv')
    df.drop(['id', 'day', 'month'], axis=1, inplace=True)
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    iso_forest = IsolationForest(n_estimators=250, contamination=0.001, random_state=42)
    iso_forest.fit(df_scaled)
    
    predictions = iso_forest.predict(df_scaled)
    df['outliers'] = predictions

    df_inliers = df[df['outliers'] == 1].copy() 

    # Plotting adjustments
    plt.figure(figsize=(18, 9))
    features = df.columns[:-1]  # exclude the 'outliers' column
    for i, feature in enumerate(features):
        # Plot all points as blue dots
        plt.scatter([feature] * len(df), df_scaled[feature], color='blue', marker='.', alpha=0.6, label='Inliers' if i == 0 else "")
        
        # Overlay outliers as red x's
        outliers_mask = df['outliers'] == -1
        plt.scatter([feature] * outliers_mask.sum(), df_scaled.loc[outliers_mask, feature], color='red', marker='x', label='Outliers' if i == 0 else "")

    plt.xticks(rotation=90)  # Rotate feature names for better readability
    plt.legend()
    plt.title('Standardized Values of Features With Outliers Marked')
    plt.xlabel('Features')
    plt.ylabel('Standardized Values')
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.savefig('outliers_per_variable.png')
    plt.show()
    
    df_inliers.drop(['outliers'], axis=1, inplace=True)
    df_inliers.to_csv('daily_aggregate_no_outliers.csv', index=False)

def remove_outliers_bystdev2():
    df = pd.read_csv('daily_aggregate_imputed.csv')
    df.drop(['id', 'day', 'month'], axis=1, inplace=True)
    df_scaled = normalise_data(df)

    # Identify outliers using 3 standard deviation rule
    outliers_mask = (np.abs(df_scaled) > 3)
    df['outliers'] = outliers_mask.any(axis=1)  # Add 'outliers' column to df

    # Plotting
    plt.figure(figsize=(18, 9))
    for i, feature in enumerate(df.columns[:-1]):  # Exclude 'outliers' column for plotting features
        # Plot inliers
        inliers = df_scaled.loc[~df['outliers'], feature]
        plt.scatter([feature] * len(inliers), inliers, color='blue', marker='.', alpha=0.6, label='Inliers' if i == 0 else "")
        
        # Plot outliers
        outliers = df_scaled.loc[df['outliers'], feature]
        plt.scatter([feature] * len(outliers), outliers, color='red', marker='x', label='Outliers' if i == 0 else "")

    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Standardized Values of Features With Outliers Marked (3 Std Dev)')
    plt.xlabel('Features')
    plt.ylabel('Standardized Values')
    plt.tight_layout()
    plt.savefig('outliers_corrected.png')
    plt.show()

    # Saving the data without outliers
    df_inliers = df[~df['outliers']].drop('outliers', axis=1)
    df_inliers.to_csv('daily_aggregate_no_outliers.csv', index=False)


def correlation_matrix():
    """Create a correlation matrix"""
    df = pd.read_csv('_normalised.csv')
    df.drop(['date'], axis=1, inplace=True)
    corr = df.corr()
    labels = [label.replace('appCat.', '') for label in corr.columns.values]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot = True,  fmt=".2f", cmap='coolwarm',
                xticklabels=labels,
                yticklabels=labels, annot_kws={"size": 10})
    plt.tight_layout()
    plt.savefig('correlation_matrix_normalised.png')
    
def box_plot_variables():
    """Create a box plot for variables"""
    df = pd.read_csv('_normalised.csv')
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)  
    plt.tight_layout()
    print('ola')
    plt.savefig('box_plot_variables.png')

def convert_date_to_day_and_month(df):
    """Convert the date column to numerical day and month columns."""
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df.drop(['date'], axis=1, inplace=True)
    return df

def impute_missing_values():
    """Impute missing values using the IterativeImputer"""
    df = pd.read_csv('daily_aggregate.csv')
    df = convert_date_to_day_and_month(df)
    df['id'] = df['id'].str.replace('AS14.', '').astype(int)
    imp = IterativeImputer(estimator=RandomForestRegressor(), max_iter=20, random_state=0)
    df_imputed = imp.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    df_imputed.to_csv('daily_aggregate_imputed.csv', index=False)

def count_noofday_per_id():
    """Count the total number of days per ID in the dataset, including days with NaN entries"""
    df = pd.read_csv('daily_aggregate.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['count'] = 1
    day_counts = df.groupby('id')['count'].sum()
    return day_counts

def compute_mahalanobis(df_scaled):
    covariance_matrix = np.cov(df_scaled.values.T)
    covariance_matrix_inverse = np.linalg.inv(covariance_matrix)
    
    mahalanobis_distances = df_scaled.apply(
        lambda row: np.sqrt(np.dot(np.dot(row, covariance_matrix_inverse), row.T)), axis=1)
    return mahalanobis_distances

def detect_outliers_mahalanobis(df_scaled, alpha=0.1):
    mahalanobis_distances = compute_mahalanobis(df_scaled)
    p_values = 1 - chi2.cdf(mahalanobis_distances, df_scaled.shape[1])
    outliers_mask = p_values < alpha
    return outliers_mask

def plot_outliers(df_scaled, outliers_mask, save_path='outliers_mahalanobis.png'):
    plt.figure(figsize=(18, 9))
    for i, feature in enumerate(df_scaled.columns):
        # Plot inliers
        inliers = df_scaled.loc[~outliers_mask, feature]
        plt.scatter([feature] * len(inliers), inliers, color='blue', marker='.', alpha=0.6, label='Inliers' if i == 0 else "")
        
        # Plot outliers
        outliers = df_scaled.loc[outliers_mask, feature]
        plt.scatter([feature] * len(outliers), outliers, color='red', marker='x', label='Outliers' if i == 0 else "")
    
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Standardized Values of Features With Outliers Marked (Mahalanobis)')
    plt.xlabel('Features')
    plt.ylabel('Standardized Values')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def remove_outliers_bystdev(df_path='daily_aggregate_imputed.csv', save_path='daily_aggregate_no_outliers.csv'):
    df = pd.read_csv(df_path)
    df.drop(['id', 'day', 'month'], axis=1, inplace=True)
    
    df_scaled = normalise_data(df)
    outliers_mask = detect_outliers_mahalanobis(df_scaled)
    plot_outliers(df_scaled, outliers_mask)
    
    df_inliers = df[~outliers_mask].copy()
    df_inliers.to_csv(save_path, index=False)

def create_sequences(data,feature_columns, n_steps=5):
    X, y = [], []
    unique_ids = data['id'].unique()

    if feature_columns is None:
        feature_columns = data.columns.difference(['id', 'mood'])

    for id_val in unique_ids:
        id_data = data[data['id'] == id_val]

        id_data = id_data.sort_values(by=['day', 'month'])

        for i in range(len(id_data) - n_steps):
            X.append(id_data.iloc[i:i+n_steps][feature_columns].values)
            y.append(id_data.iloc[i + n_steps]['mood'])
    
    return np.array(X), np.array(y)


def train_LSTM_regressor(X_train, y_train, epochs=1):
    model = Sequential([
        LSTM(100, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    seqno = 1

    for sequence, target in zip(X_train, y_train):
        batch_size = len(sequence)
        
        sequence = np.expand_dims(sequence, axis=0)
        target = np.expand_dims(target, axis=0)

        print(f"Training for sequence: {seqno}")
        seqno += 1
        model.fit(sequence, target, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    
    return model

def train_LSTM_classifier(X_train, y_train, epochs=1):
    model = Sequential([
        LSTM(100, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    seqno = 1

    for sequence, target in zip(X_train, y_train):
        batch_size = len(sequence)
        
        sequence = np.expand_dims(sequence, axis=0)
        target = np.expand_dims(target, axis=0)

        print(f"Training for sequence: {seqno}")
        seqno += 1
        model.fit(sequence, target, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    
    return model



def load_and_preprocess_data(filepath, type):
    df = pd.read_csv(filepath)

    if type == 'classification':
        df['mood'] = df['mood'].round(0).astype(int)
    
    train_data = []
    test_data = []

    for id_val in df['id'].unique():
        id_data = df[df['id'] == id_val]
        id_data = id_data.sort_values(by=['month', 'day'])

        train_size = int(0.8 * len(id_data))
        train_id_data = id_data.iloc[:train_size]
        test_id_data = id_data.iloc[train_size:]
        
        train_data.append(train_id_data)
        test_data.append(test_id_data)

    train_data = pd.concat(train_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)

    return train_data, test_data

def run_LSTM(feature_columns, type, filepath='daily_aggregate_imputed.csv'):
    train_data, test_data = load_and_preprocess_data(filepath, type)
    
    X_train, y_train = create_sequences(train_data, feature_columns)
    X_test, y_test = create_sequences(test_data, feature_columns)

    print(X_train)
    if type == 'classification':
        model = train_LSTM_classifier(X_train, y_train)
        y_pred = lstm_classification_prediction(model, X_test)
    elif type == 'regression':
        model = train_LSTM_regressor(X_train, y_train)
        y_pred = lstm_regression_prediction(model, X_test)

    model.save(f'lstm_{type}_2epoch.keras')
    
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {mse}")

    plotpredictions(y_pred, y_test, type)

def lstm_regression_prediction(model, Xtest):
    y_pred = model.predict(Xtest)
    return y_pred.flatten()

def lstm_classification_prediction(model, Xtest):
    y_pred = model.predict(Xtest)
    return y_pred.argmax(axis=1)

def plotpredictions(y_pred, y_test, type):
    plt.figure(figsize=(10, 6))
    if type == 'RandomForestClassifier':
        plt.scatter(range(len(y_test)), y_test, color='blue', label='True Mood', alpha=0.5, edgecolors='w')
        plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Mood', alpha=0.5, edgecolors='w')
    else:
        plt.plot(y_test, label='True Mood', color='blue')
        plt.plot(y_pred, label='Predicted Mood', color='red')
    plt.legend()
    plt.grid()
    plt.title('True vs Predicted Mood')
    plt.savefig(f'{type}_true_vs_predicted_mood.png')

def create_features(df, feature_columns, n_lags=5):
    """Extend DataFrame with lagged features of interest."""
    lagged_df = pd.DataFrame()
    for column in feature_columns:
        for lag in range(1, n_lags + 1):
            lagged_df[f'{column}_lag{lag}'] = df[column].shift(lag)
    lagged_df['mood'] = df['mood']
    lagged_df.dropna(inplace=True)
    return lagged_df[lagged_df.columns.difference(['mood'])], lagged_df['mood']

def random_forest_classifier(filepath, feature_columns, n_estimators=100, random_state=42):
    """Train a Random Forest Classifier on the dataset."""
    df = pd.read_csv(filepath)

    df['mood'] = df['mood'].round(0).astype(int)
    
    X, y = create_features(df, feature_columns, n_lags=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test MSE: {mse}")
    print(f"Test Accuracy: {accuracy}")

    plotpredictions(y_pred, y_test, type='RandomForestClassifier')

def build_lstm_hypermodel(X_train, y_train, hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='tanh',
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=False
    ))
    model.add(Dense(
        10, activation='softmax'
    ))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    feature_columns = ['activity', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.weather', 'circumplex.valence']
    #wide_df = prepare_data_for_correlation(df)
    #wide_df.to_csv('wide_df.csv', index=False)
    #create_correlation_matrix(wide_df)
    #sorted_data()
    #average_through_ids()
    #normalised_data()
    #correlation_matrix()
    #box_plot_variables()
    #plot_attribute_over_time_comparison('mood')
    #make_daily_aggragate_csv()
    #replace_negative_vals()
    #impute_missing_values()
    #make_daily_aggregate_csv()
    #print(count_noofday_per_id())
    #plot_dailyavg_variable_distribution('mood')
    #remove_outliers_forest()
    #run_LSTM(feature_columns, type='classification', filepath='daily_aggregate_imputed.csv')
    random_forest_classifier('clean_data_with_columns.csv', feature_columns)