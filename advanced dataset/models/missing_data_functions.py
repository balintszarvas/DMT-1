''' 
This file contains the missing data functions
'''

import numpy as np
import pandas as pd




def forward_filling(df_or):
    """
    Does forward filling and then backward filling only for values that have neighbors.
    If there are any extra missing, it computes the average
    """
    df = df_or.copy()
    
    df_filled = df.groupby('id').ffill(limit = 1).bfill(limit = 1)
    common_columns = df.columns.intersection(df_filled.columns)
    
    df[common_columns] = df_filled

    df = average(df)

    return df

def average(df_or):
    """
    Fills the missing values by the average of all values on that day, then average of id and the extra missing values by the total average
    """
    df = df_or.copy()
    df["appCat.unknown"] = pd.to_numeric(df["appCat.unknown"], errors='coerce')
    
    
    date_means = df.groupby('date').mean(numeric_only = True)
    columns_to_fill = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns_to_fill:
        nan_indices = df[col].isna()
        df.loc[nan_indices, col] = df.loc[nan_indices, 'date'].map(date_means[col])
    
    id_means = df.groupby('id').mean(numeric_only = True)
    columns_to_fill = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns_to_fill:
        nan_indices = df[col].isna()
        df.loc[nan_indices, col] = df.loc[nan_indices, 'id'].map(id_means[col])

    df = df.fillna(df.mean(numeric_only= True))

    return df

def interpolation(df_or):
    """
    Does linear interpolations, values that are not interpolated then are approximated with the average
    """
    df = df_or.copy()
    df["appCat.unknown"] = pd.to_numeric(df["appCat.unknown"], errors='coerce')
    ids = df['id'].unique()
    
    for id_ in ids:
        data = df[df['id'] == id_]
        data.set_index('date', inplace=True)
        interpolated_data = data.select_dtypes(include=[np.number]).interpolate(method = 'linear')
        df.loc[df['id'] == id_, interpolated_data.columns] = interpolated_data.values
    
    df = average(df)
    return df
