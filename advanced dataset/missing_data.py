""" This file contains the function for making the prediction models"""


import numpy as np
import pandas as pd
import seaborn as sns

from scipy.linalg import svd

import matplotlib.pyplot as plt

from sklearn import datasets, model_selection
from models.missing_data_functions import forward_filling, average, interpolation




class Missing_data():

    def __init__(self, df):
        """

        This class is made for missing the missing values of the data. 

        Inputs:

            - df: Dataframe,Assumes that dataset is ordered by time and of the form:
                id,date,variables columns

        Variables:

            - df: Dataframe which will change

            - df_or: Copy of the original Dataframe
        """

        self.df = df
        self.df_or = df.copy()
        self.nan_var = None
        self.nan_count = None

        self.ids = self.df['id'].unique() 
        self.date = self.df['date'].unique() 

        self.mapping = {
            'average': average,
            'interpolation': interpolation,
            'forward filling': forward_filling,
        }


    def nan_finder(self):
        """ 
        Given the data frame, returns the variables with NaN values, the number of NaN Values and it's index
        """
        
        self.nan_var = self.df.columns[self.df.isnull().any()].tolist()
        self.nan_count = self.df.isna().sum().sum()
        self.nan_positions = np.where(pd.isnull(df))
        self.nan_indices = [[self.df.index[self.nan_positions[1]].tolist()],self.df.columns[self.nan_positions[1]].tolist()]

    def fill_missing(self, mode = 'average', info = True):
        """
        Fills the missing values with a specified method.
        Inputs:
        - mode: mode for filling, string of:average_mode, moving_average, interpolation
        """
        self.nan_finder()
        initial_nan = self.nan_count
        method = self.mapping.get(mode)
        
        if self.nan_count == 0:
            print("The dataset is already full")
        else:
            self.df = method(self.df_or)
            self.nan_finder()
            print(f"Filled {initial_nan - self.nan_count} values and {self.nan_count} values are still missing")
            
            if info == True:
                print("ORIGINAL UNFILLED DATA")
                print(self.df_or.isna().sum())
                print("UNFILLED DATA")
                print(self.df.isna().sum())
                self.plot_missing(mode)

    def plot_missing(self,mode):
            """ 
            Saves tow figures, one of teh original dataset missing data, and one of the old dataset missing data
            """

            unfilled = self.df.groupby('id').apply(lambda x: x.iloc[:, 1:].isna().sum())
            filled = self.df.groupby('id').apply(lambda x: x.iloc[:, 1:].notna().sum())

            unfilled = unfilled/(filled + unfilled)
            
            unfilled_or = self.df_or.groupby('id').apply(lambda x: x.iloc[:, 1:].isna().sum())
            filled_or = self.df_or.groupby('id').apply(lambda x: x.iloc[:, 1:].notna().sum())
            unfilled_or = unfilled_or/(filled_or + unfilled_or)


            plt.figure(figsize=(15, 15))
            sns.heatmap(unfilled, annot=True, fmt=".2f", cmap='coolwarm',
                        xticklabels=unfilled.columns.values,
                        yticklabels=unfilled.index)

            plt.tight_layout()
            plt.grid(True)
            plt.savefig(f'advanced dataset/images/new_filled_with_{mode}.png')

            plt.figure(figsize=(15, 15))

            sns.heatmap(unfilled_or, annot=True, fmt=".2f", cmap='coolwarm',
                        xticklabels=unfilled.columns.values,
                        yticklabels=unfilled.index)

            plt.tight_layout()
            plt.grid(True)
            plt.savefig('advanced dataset/images/old_filled.png')


    def reset(self):
        """
        Resets the current dataframe to its original form, without filled missed values
        """

        self.df = self.df_or.copy()

    

    def save(self, name = 'Unnamed.csv', reset = False):
        """

        saves the current dataframe:

        Inputs:

            - name: Name of the file, if not specified will overwrite the Unnamed file. Data is always stored in missing values dataframe

            - reset: Once saved, resets the current dataframe to it's original form, default is True
        """
        
        self.df.to_csv(f"advanced dataset/data/{name}", index = False)
        
        if reset == True:
            self.reset()





if __name__ == '__main__':

    df = pd.read_csv('../DMT-1/daily_aggregate.csv')

    mis_df = Missing_data(df)
    mis_df.fill_missing(mode = 'forward filling', info= True)



    



    
    




