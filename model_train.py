# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:46:01 2024

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
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer

from data_prep import prepare_data


df = pd.read_csv("dataset.csv")
df = prepare_data(df)




numeric_features = ['Year', 'Hand','capacity_Engine', 'Km','Pic_num']  # רשימת העמודות המספריות
categorical_features = ['Gear', "model",'manufactor', 'Engine_type']  # רשימת העמודות הקטגוריות

# Preprocessing step (OneHotEncoder and StandardScaler)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)     ])

# Model training step (ElasticNet with RandomizedSearchCV)
model_step = RandomizedSearchCV(
    estimator=ElasticNet(),
    param_distributions={
        'alpha': uniform(loc=0.001, scale=0.1),  # Uniform distribution for alpha
        'l1_ratio': uniform(loc=0.0, scale=1.0)   # Uniform distribution for l1_ratio
    },
    n_iter=5,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42
)

# Create the full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_step)])
X = df.drop(columns="Price")
y = df['Price']

# Remove rows with NaN values (optional)
X = X.dropna()
y = y[X.index]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model's performance
# (Use metrics like mean squared error or R-squared)

# חישוב שגיאת ממוצע הריבועים (MSE)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Best Parameters: {best_params}")
print(f"Mean Squared Error: {mse}")
print(f'RMSE: {rmse}')
print(f"R-squared Score: {r2}")

import pickle
pickle.dump(pipeline,open("model.pkl","wb"))



