import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
import csv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2

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

    date_column = df['date']
    df = df.drop(columns='date')

    df.replace("", np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    mean = df.mean()
    std = df.std()

    df = (df - mean) / std
    df = pd.concat([date_column, df], axis=1)
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
    sns.heatmap(corr, fmt=".2f", cmap='coolwarm',
                xticklabels=labels,
                yticklabels=labels)
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


if __name__ == '__main__':
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
    remove_outliers_forest()