""" This file contains the function for making the prediction models"""

import numpy as np
import pandas as pd

from scipy.linalg import svd

import matplotlib.pyplot as plt

from sklearn import datasets, model_selection




class Missing_data():


    def __init__(self, df):
        """

        This class is made for missing the missing values of the data

        Inputs:

            - df: Dataframe

        Variables:

            - df: Dataframe which will change

            - df_or: Copy of the original Dataframe
        """
        
        df = pd.read_csv('C:/Users/joann/OneDrive/Escritorio/Intro/DMT-1/dataset_mood_smartphone.csv')
        df = df.drop('Unnamed: 0', axis = 1)
        df[['day', 'time']] = df['time'].str.split(' ', expand=True)
        self.df = df
        self.df_or = df.copy()
        self.nan_var = None
        self.nan_count = None
        self.ids = self.df['id'].unique() 


    def nan_finder(self):
        """ 

        Given the data frame, returns the variables with NaN values and the number of Nan Values
        """
        
        self.nan_var = self.df[self.df.isna().any(axis=1)]['variable'].unique()
        self.nan_count = self.df.isna().sum().sum()
        self.nan_indices = self.df.index[df['value'].isna()]



    def moving_average(self):
        """ 

        Given the data frame, fills the columns that are empty with the ID moving average
        """
        self.nan_finder()  

        for i_d in self.ids:
            for var in self.nan_var:
                data = df[(df['id'] == i_d) & (df['variable'] == var)]['value']
                
                nan_index = data.index[data.isna()]
        
                nan_positions = np.where(data.isna())
                
                for i,pos in enumerate(nan_positions[0]):
                    mean = data[:pos].mean()
                    self.df.loc[nan_index[i],'value'] = mean
        
        miss_df.nan_finder()
        miss_df.df.loc[miss_df.nan_indices[0],'value'] = 0

    def plotter(self,var):
        """
        PLots the evolution of a dataset overtime of one variable.
        Inputs:
            - Variable: Variable to plot
        """
        for id_ in self.ids:
            
            data_id = self.df[self.df['id'] == id_]
            
            plt.figure(figsize=(10, 6))
            plt.plot(data_id['time'], data_id[data_id['variable'] == var]['values'], label=f'ID {id_}')
    
            plt.title(f'{var} Over Time (ID {id_})')
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            plt.show()
            

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
        
        self.df.to_csv(name, index = False)
        

        if reset == True:
            self.reset()


if __name__ == '__main__':

    df = pd.read_csv('../DMT-1/dataset_mood_smartphone.csv')
    df = df.drop('Unnamed: 0', axis = 1)
    df[['day', 'time']] = df['time'].str.split(' ', expand=True)
 

    miss_df = Missing_data(df)
    miss_df.nan_finder()

    miss_df.moving_average()

    



    
    




