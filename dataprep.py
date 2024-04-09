import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
import csv

file_path = "dataset_mood_smartphone.csv"
df = pd.read_csv(file_path)

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
    booleans = ["SMS", "call"]
    for id in id_list:
        for variable in booleans:
            df[(df['id'] == id) & (df['variable'] == variable)].groupby('time').agg('sum')


def count_nan_values(df):
    """Count the number of NaN values in the dataset"""
    nan_values = df.isna().sum()
    return nan_values

def prepare_data_for_correlation(df):
    """Prepare the data for correlation analysis by converting it to wide format"""
    df_wide = df.pivot_table(index=['id', 'time'], columns='variable', values='value')
    df_wide.reset_index(inplace=True)
    # Now drop the non-numeric columns 'id' and 'time', if you want to include 'time' convert it to a numeric value first
    df_wide.drop(['id', 'time'], axis=1, inplace=True)
    return df_wide

def create_correlation_matrix(df_wide):
    """Create a correlation matrix for the wide format data"""
    corr = df_wide.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('correlation_matrix.png')

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

def plot_mood_over_time_comparison():
    """Plot the mood over time in the dataset, comparing minute-level data to daily averages."""
    # Filter for 'mood' variable
    df_mood = df[df['variable'] == 'mood']
    df_mood['time'] = pd.to_datetime(df_mood['time'])

    # Plotting mood values for each timestamp
    df_minute_avg = df_mood.groupby(['time']).agg({'value': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    plt.plot(df_minute_avg['time'], df_minute_avg['value'], label='Mood over Time')

    # Computing daily average of mood values
    df_mood['date'] = df_mood['time'].dt.date
    df_daily_avg = df_mood.groupby(['date']).agg({'value': 'mean'}).reset_index()
    df_daily_avg['date'] = pd.to_datetime(df_daily_avg['date'])  # Converting 'date' back to datetime for plotting

    # Plotting the daily average trajectory
    plt.plot(df_daily_avg['date'], df_daily_avg['value'], 'r', label='Daily Average Mood')

    # Adding plot details
    plt.ylabel('Mood')
    plt.title('Mood over Time and Daily Average Mood')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mood_over_time_comparison.png')


def plot_dailyavg_mood_distribution():
    """Plot the daily average mood distribution in the dataset"""
    df_mood = df[df['variable'] == 'mood']
    df_mood['time'] = pd.to_datetime(df_mood['time'])
    df_mood['date'] = df_mood['time'].dt.date
    df_mood = df_mood.groupby(['date']).agg({'value': 'mean'}).reset_index()
    plt.figure(figsize=(10, 8))
    sns.distplot(df_mood['value'], bins=100, kde=False)
    plt.xlabel('Mood')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.title('Daily Average Mood Distribution')
    plt.savefig('dailyavg_mood_distribution.png')

def make_daily_aggragate_csv():
    """Create a CSV file with daily aggregate values for each variable (average for real valued, sum for boolean)"""
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df.drop(['time'], axis=1, inplace=True)
    df_boolean = df[df['variable'].isin(['call', 'sms'])]
    df_real = df[~df['variable'].isin(['call', 'sms'])]
    df_boolean = df_boolean.groupby(['id', 'date', 'variable']).agg({'value': 'sum'}).reset_index()
    df_real = df_real.groupby(['id', 'date', 'variable']).agg({'value': 'mean'}).reset_index()
    df_aggregate = pd.concat([df_boolean, df_real])
    df_aggregate = df_aggregate.pivot_table(index=['id', 'date'], columns='variable', values='value').reset_index()
    df_aggregate.to_csv('daily_aggregate.csv', index=False)
    return df_aggregate

def average_through_ids():
    """Averge the daily values of variables through IDs"""
    df = pd.read_csv('daily_aggregate.csv')
    df = df.groupby('date').agg('mean').reset_index()
    df.to_csv('daily_aggregate_avg.csv', index=False)
    
def normalized_data():
    """Normalize the data"""
    df = pd.read_csv('daily_aggregate_avg.csv')

    date_column = df['date']
    df = df.drop(columns='date')

    df.replace("", np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    mean = df.mean()
    std = df.std()

    df = (df - mean) / std
    df = pd.concat([date_column, df], axis=1)
    df.to_csv('daily_aggregate_normalized.csv', index=False)
    
def correlation_matrix():
    """Create a correlation matrix"""
    df = pd.read_csv('daily_aggregate_normalized.csv')
    df.drop(['date'], axis=1, inplace=True)
    corr = df.corr()
    labels = [label.replace('appCat.', '') for label in corr.columns.values]

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, fmt=".2f", cmap='coolwarm',
                xticklabels=labels,
                yticklabels=labels)
    plt.tight_layout()
    plt.savefig('correlation_matrix_normalized.png')
    
def box_plot_variables():
    """Create a box plot for variables"""
    df = pd.read_csv('daily_aggregate_normalized.csv')
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)  
    plt.tight_layout()
    plt.savefig('box_plot_variables.png')



if __name__ == '__main__':
    #wide_df = prepare_data_for_correlation(df)
    #wide_df.to_csv('wide_df.csv', index=False)
    #create_correlation_matrix(wide_df)
    #average_through_ids()
    normalized_data()
    correlation_matrix()
    box_plot_variables()