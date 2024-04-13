import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
import csv
from statsmodels.tsa.seasonal import seasonal_decompose

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
    
def normalised_data():
    """Normalize the data"""
    # Read the dataset
    df = pd.read_csv('daily_aggregate.csv')

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
    
def correlation_matrix():
    """Create a correlation matrix"""
    df = pd.read_csv('daily_aggregate_normalised.csv')
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
    df = pd.read_csv('daily_aggregate_normalised.csv')
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)  
    plt.tight_layout()
    print('ola')
    plt.savefig('box_plot_variables.png')

def variables_distribution_plot():
    """Create a distribution plot for variables"""
    df = pd.read_csv('daily_aggregate_normalised.csv')
    df.drop(['date'], axis=1, inplace=True)
    df = pd.melt(df)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='variable', y='value', data=df)
    plt.xticks(rotation=90)  
    plt.tight_layout()
    plt.savefig('variables_distribution_plot.png')
    

def important_distribution_plot():
    """Create a distribution plot for mood"""
    df = pd.read_csv('daily_aggregate_normalised.csv')
    df.drop(['date'], axis=1, inplace=True)
    sns.set(style="whitegrid")
    important = ['mood', 'circumplex.valence', 'activity', 'appCat.communication', 'call', 'screen']
    for i in important:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=i, data=df)
        plt.grid(True, which='both')  
        plt.ylabel('Density') 
        plt.tight_layout()
        plt.savefig(f'{i}_distribution_plot.png')

def plot_range_hours_ID(df):
    df['time'] = pd.to_datetime(df['time'])
    
    plt.figure()
    plt.title('Scatter plot for all IDs')
    plt.xlabel('ID')
    plt.ylabel('Hour')
    plt.xticks(rotation=45)  
    
    unique_ids = df['id'].unique()
    for id in unique_ids:
        id_data = df[df['id'] == id]
        hours = id_data['time'].dt.strftime('%H:%M')
        plt.scatter([id] * len(hours), hours)
    
    # Manually set y-axis tick locations and labels every hour
    y_ticks = pd.date_range(start='00:00', end='23:59', freq='1H').strftime('%H:%M')
    plt.yticks(y_ticks)
    
    plt.show()



if __name__ == '__main__':
    #wide_df = prepare_data_for_correlation(df)
    #wide_df.to_csv('wide_df.csv', index=False)
    #create_correlation_matrix(wide_df)
    plot_mood_over_time_comparison()
    #average_through_ids()
    #normalised_data()
    #correlation_matrix()
    #box_plot_variables()
    #variables_distribution_plot()
    #important_distribution_plot()