""" This File contains SVD model for dimensionality reduction"""
import numpy as np
import pandas as pd


def normalised_data(df):
    """Normalize the data
    Inputs:
        - df: Dataframe already filled data with missing values
    Outputs:
        - df: df with data normalized
    """

    date_column = df['date']
    id_column = df['id']

    df.drop(columns=['date', 'id'], inplace = True)
  


    mean = df.mean()
    std = df.std()

    df = (df - mean) / std
    df = pd.concat([id_column,date_column, df], axis=1)
    return df


def svd(df, keep = 0.9):
    """
    Does svd to a dataset.
    Inputs:
        - df: data frame, with structure, id,data,variables
        - keep: How many data/information to keep, default is 90%
    Outputs:
        - df: Dataframe with applied dimensionality reduction
    """
    df = normalised_data(df)

    date_column = df['date']
    id_column = df['id']

    df.drop(columns=['date', 'id'], inplace = True)
    df = df.apply(pd.to_numeric, errors='coerce')
    U, s, Vt = np.linalg.svd(df, full_matrices=False)
    
    total_s_squared = np.sum(s**2)
    cumulative_s = np.cumsum(s**2) / total_s_squared
    
    num_components = np.argmax(cumulative_s>= keep) + 1

    U_reduced = U[:, :num_components]
    s_reduced = s[:num_components]
    Vt_reduced = Vt[:num_components, :]

    df_reduced = pd.DataFrame(np.dot(U_reduced, np.diag(s_reduced)))
    df_reduced['date'] = date_column
    df_reduced['id'] = id_column

    return df_reduced, df_aprox



if __name__ == '__main__':
    df = pd.read_csv('../DMT-1/advanced dataset/data/train.csv')
    print(df.columns)
    df,df_aprox = svd(df)
    print(df.columns)


