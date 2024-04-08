import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import path, makedirs
import csv

file_path = "dataset_mood_smartphone.csv"
df = pd.read_csv(file_path)

def count_id(df):
    id_count = df['id'].nunique()
    return id_count

def get_id(df):
    id_list = df['id'].unique()
    return id_list

def get_variables(df, variables):
    variable_list = df['variable'].isin(variables)
    return variable_list

def timestep_per_id_count(df, variables):
    id_list = get_id(df)
    timestep_count = []
    for id in id_list:
        for variable in variables:
            timestep_count.append(len(df[(df['id'] == id) & (df['variable'] == variable)]))
    return timestep_count

def plot_scatter_timesteps_per_id(df, variables):
    id_list = get_id(df)
    variable_list = get_variables(df, variables)
    timestep_count = timestep_per_id_count(df, variables)
    plt.scatter(id_list, timestep_count)
    plt.xlabel('ID')
    plt.ylabel('Timestep Count')
    plt.title('Timestep Count per ID')
    plt.savefig('timestep_count_per_id.png')

def prepare_data_for_correlation(df):
    df_wide = df.pivot_table(index=['id', 'time'], columns='variable', values='value')
    df_wide.reset_index(inplace=True)
    # Now drop the non-numeric columns 'id' and 'time', if you want to include 'time' convert it to a numeric value first
    df_wide.drop(['id', 'time'], axis=1, inplace=True)
    return df_wide

def create_correlation_matrix(df_wide):
    corr = df_wide.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('correlation_matrix.png')


if __name__ == '__main__':
    wide_df = prepare_data_for_correlation(df)
    wide_df.to_csv('wide_df.csv', index=False)
    create_correlation_matrix(wide_df)