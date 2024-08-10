# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:41:25 2024

@author: USER
"""

import pandas as pd  # ייבוא pandas לעבודה עם מסגרות נתונים
import matplotlib.pyplot as plt  # ייבוא matplotlib ליצירת גרפים
import numpy as np  # ייבוא numpy לחישובים מתמטיים
import urllib.request, urllib.parse, urllib.error  # ייבוא ספריות לעבודה עם בקשות רשת
import requests  # ייבוא requests לשליחת בקשות HTTP
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


df = pd.read_csv("dataset.csv")
# df




na_per_column = df.isnull().sum(axis=0)
na_per_column


# Function to convert string numbers with commas to float
def convert_to_float(value):
    if value is None or value == 'None':
        return float("nan")
    if isinstance(value, str):
        return float(value.replace(',',''))
# Function to replace low values with the mean of the same manufactor and model
def replace_with_mean(group):
    threshold = 600
    # Calculate mean excluding 0 and values below threshold
    mean_value = group[(group['capacity_Engine'] >= threshold)]['capacity_Engine'].mean()
    # Replace values below threshold with the mean, ignoring 0
    group.loc[(group['capacity_Engine'] < threshold) & (group['capacity_Engine'] != 0), 'capacity_Engine'] = mean_value
    return group

## This function imputes missing values using KNN for several columns:
##  ['Engine_type','Year','Hand','Km','Pic_num','capacity_Engine']
def Fill_Engine_type(df):
    # Initialize LabelEncoder for categorical columns
    label_encoder = LabelEncoder()
    
    # Encode categorical columns to numerical values
    df['Engine_type_encoded'] = label_encoder.fit_transform(df['Engine_type'].astype(str))
    
    # Columns to impute
    impute_columns = ['Year','Hand','Km','Pic_num','capacity_Engine', 'Engine_type_encoded']
    
    # Create KNNImputer object
    imputer = KNNImputer(n_neighbors=2)
    
    # Impute missing values
    df[impute_columns] = imputer.fit_transform(df[impute_columns])
    
    # Convert encoded values back to categorical
    df['Engine_type'] = label_encoder.inverse_transform(df['Engine_type_encoded'].astype(int))
    
    # Drop the temporary encoded columns
    df.drop(columns=['Engine_type_encoded'], inplace=True)
    
    return df

# This function impute missing values after encoded columns : manufactor,gear,model.
def Fill_Gear(df):
    # Encode 'manufactor' and 'model' columns
    label_encoder_manufactor = LabelEncoder()
    df['manufactor_encoded'] = label_encoder_manufactor.fit_transform(df['manufactor'].astype(str))

    label_encoder_model = LabelEncoder()
    df['model_encoded'] = label_encoder_model.fit_transform(df['model'].astype(str))

    # Encode 'Gear' column
    label_encoder_gear = LabelEncoder()
    df['Gear_encoded'] = label_encoder_gear.fit_transform(df['Gear'].astype(str))

    # Create KNNImputer object
    imputer = KNNImputer(n_neighbors=2)

    # Impute missing values for encoded columns
    impute_columns = ['manufactor_encoded', 'model_encoded', 'Gear_encoded']
    df[impute_columns] = imputer.fit_transform(df[impute_columns])

    # Convert encoded values back to categorical
    df['manufactor'] = label_encoder_manufactor.inverse_transform(df['manufactor_encoded'].astype(int))
    df['model'] = label_encoder_model.inverse_transform(df['model_encoded'].astype(int))
    df['Gear'] = label_encoder_gear.inverse_transform(df['Gear_encoded'].astype(int))

    # Drop the temporary encoded columns
    df.drop(columns=['manufactor_encoded', 'model_encoded', 'Gear_encoded'], inplace=True)

    return df

def prepare_data(df):
    
    # Drop rows with NA in 'Price'
    df = df.dropna(subset=['Price'])
    # remove columns that not neccesery 
    columns_to_drop = ['Prev_ownership','Curr_ownership','Color','Area','City', 'Cre_date', 'Repub_date', 'Description', 'Supply_score','Test']
    df = df.drop(columns=columns_to_drop)
    
    # Convert 'Year', 'Km','capacity_Engine' and 'Pic_num' to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Km'] = df['Km'].apply(convert_to_float)
    df['capacity_Engine'] = df['capacity_Engine'].apply(convert_to_float)
    df['Pic_num'] = df['Pic_num'].apply(convert_to_float)
    
    # If Pic_num is NA fill with 0
    df['Pic_num'] = df['Pic_num'].fillna(0)    # Correcting values of Km column: multiplying values less than 1000 by 1000
    df['Km'] = df['Km'].apply(lambda x: x*1000 if x < 1000 else x)
    
    # Replace special values
    df['Gear'] = df['Gear'].replace({'אוטומט': 'אוטומטית', 'לא מוגדר': np.nan})
    # These scripts will replace all instances of "היבריד" with "היברידי" in the Engine_type column  
    df['Engine_type'] = df['Engine_type'].replace('היבריד', 'היברידי')
    # These scripts will replace all instances of "Lexsus" with "לקסוס" in the manufactor column  
    df['manufactor'] = df['manufactor'].replace('לקסוס','Lexsus')
    # Change the Engine type of the electric car to zero     
    df.loc[df['Engine_type'] == 'חשמלי', 'capacity_Engine'] = 0
    
    # Calculate mean Km by Year
    mean_km_by_year = df.groupby('Year')['Km'].mean()
    # Replace zero Km values with mean values based on Year
    df['Km'] = df.apply(lambda row: mean_km_by_year[row['Year']] if row['Km'] == 0 else row['Km'], axis=1)
    
    # Filter rows where capacity_Engine > 10000 and divide those values by 10 outliers
    df.loc[df['capacity_Engine'] > 10000, 'capacity_Engine'] /= 10
    
    # Apply the function to fillNA at catagorical col(Engine_type) and numerical columns . 
    df = Fill_Engine_type(df)
    
    # Impute missing values after making catagorical columns encoded:
    # ['manufactor','gear','model']
    df = Fill_Gear(df)
    
    ### one hot encoder for catagorical columns: 
    # Apply preprocess_data func on our data- to make One Hot Encoder
    #df = preprocess_data(df)
     
    # Define a threshold for low engine capacity values
   # threshold = 600

    # Find all columns related to manufactor and model
   # manufactor_columns = [col for col in df.columns if col.startswith('manufactor_')]
    #model_columns = [col for col in df.columns if col.startswith('model_')]

    # Combine manufactor and model columns to group by
    group_columns = ["manufactor", "model"]

    # Group by manufactor and model columns and apply the replacement function
    df = df.groupby(group_columns).apply(replace_with_mean, include_groups=False).reset_index()

    return df













