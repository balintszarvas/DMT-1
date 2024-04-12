''' 
This file contains the missing data functions
'''

import numpy as np
import pandas as pd




def forward_filling(df_or):
    """
    Does forward filling and then backward filling. Lastly, the missing values are filled with the average
    """
    df = df_or.copy()
    
    df_filled = df.groupby('id').ffill().bfill()
    common_columns = df.columns.intersection(df_filled.columns)
    
    df[common_columns] = df_filled

    return df

def moving_average(df_or):
    """ 
    Given the data frame, fills the columns that are empty with the moving average, and then with the average
    """
    df = df_or.copy()
    
    for i_d in self.ids:
        data = df[(df['id'] == i_d)]
        cumulative_means = data.expanding().mean()
        data = data.fillna(cumulative_means)
        print(data)
        df.loc[df['id'] == i_d] = data
      
    df = average(df)

    return df

def average(df_or):
    """
    Fills the missing values by the average of all values on that day, then average of id and the extra missing values by the total average
    """
    df = df_or.copy()
    print(df["appCat.unknown"])
    date_means = df.groupby('date').mean(numeric_only = True)
    id_means = df.groupby('id').mean(numeric_only = True)
    columns_to_fill = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns_to_fill:
        nan_indices = df[col].isna()
        df.loc[nan_indices, col] = df.loc[nan_indices, 'date'].map(date_means[col])
    
    columns_to_fill = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in columns_to_fill:
        nan_indices = df[col].isna()
        df.loc[nan_indices, col] = df.loc[nan_indices, 'id'].map(id_means[col])


    print(df)
    df = df.fillna(df.mean(numeric_only= True))

    return df

def interpolation(df):
    df = df_or.copy()
    return None
