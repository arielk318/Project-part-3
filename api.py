import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
from data_prep import prepare_data
import joblib

app = Flask(__name__)


# טעינת המודל ושמות התכונות
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    relevant_columns = ['manufactor', 'model', 'Year', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Pic_num',
                        'Km']
    input_data = {col: [np.nan] for col in relevant_columns}

    input_data.update({
        'manufactor':  [request.form.get('manufactor', np.nan)],
        'model':   [request.form.get('model', np.nan)],
        'Year':[int(request.form.get('Year', np.nan)) if request.form.get('Year') else np.nan],
        'Hand': [request.form.get('Hand', np.nan)],
        'Gear': [request.form.get('Gear', np.nan)],
        'capacity_Engine': [
            int(request.form.get('capacity_Engine', np.nan)) if request.form.get('capacity_Engine') else np.nan],
        'Engine_type': [request.form.get('Engine_type', np.nan)],
        'Pic_num':[int(request.form.get('Pic_num', np.nan)) if request.form.get('Pic_num') else np.nan],
        'Km': [int(request.form.get('Km', np.nan)) if request.form.get('Km') else np.nan]
    })

    df = pd.DataFrame(input_data)

    # חיזוי
    prediction = model.predict(df)

    return render_template('index.html', prediction_text=f"{prediction[0]:,}₪")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

