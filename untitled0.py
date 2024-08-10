# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:12:00 2024

@author: USER
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
import joblib


model = joblib.load('model.pkl')
relevant_columns = ['manufactor', 'model', 'Year', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Pic_num',
                        'Km']
input_data = {col: [np.nan] for col in relevant_columns}

input_data.update({
    'manufactor': "טויוטה", # [request.form.get('manufactor', np.nan)],
    'model':  "קורולה" , # [request.form.get('model', np.nan)],
    'Year':2015 ,#[int(request.form.get('Year', np.nan)) if request.form.get('Year') else np.nan],
    'Hand': 1 ,# [request.form.get('Hand', np.nan)],
    'Gear': "אוטומטית", #[request.form.get('Gear', np.nan)],
    'capacity_Engine':1600,# [
        #int(request.form.get('capacity_Engine', np.nan)) if request.form.get('capacity_Engine') else np.nan],
    'Engine_type':"בנזין" ,# [request.form.get('Engine_type', np.nan)],
    'Pic_num': 2 ,#[int(request.form.get('Pic_num', np.nan)) if request.form.get('Pic_num') else np.nan],
    'Km':150000,# [int(request.form.get('Km', np.nan)) if request.form.get('Km') else np.nan]
})

    
    
df = pd.DataFrame(input_data, index=range(len(input_data)))  # Creates an index from 0 to len(input_data)-1

prediction = model.predict(df)[0]

    
    
    
    
    
    
