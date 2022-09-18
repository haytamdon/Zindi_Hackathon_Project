import os
import json
from flask import Flask, request
import pandas as pd
import pickle

def load_model():
    loaded_model = pickle.load(open('Zindi_Hackathon_Project\deployment\clf.sav', 'rb'))
    return loaded_model

def run_inference(data):
    loaded_model = load_model()
    y_pred = loaded_model.predict(data)
    return y_pred

def post_process(output):
    return str(output)

app = Flask(__name__)

@app.route("/test", methods=['POST'])
def test():
    Sensor1 = request.files['Sensor1_PM2.5'].read()
    Sensor2 = request.files['Sensor2_PM2.5'].read()
    Temperature = request.files['Temperature'].read()
    Relative_Humidity = request.files['Relative_Humidity'].read()
    Datetime_day = request.files['Datetime_day'].read()
    Datetime_month = request.files['Datetime_month'].read()
    Datetime_year = request.files['Datetime_year'].read()
    Datetime_hour = request.files['Datetime_hour'].read()
    print(Sensor1)
    L = [[float(Sensor1), float(Sensor2), float(Temperature), float(Relative_Humidity), float(Datetime_day), float(Datetime_month), float(Datetime_year), float(Datetime_hour)]]
    data = pd.DataFrame(L, columns=('Sensor1_PM2.5', 'Sensor2_PM2.5', 'Temperature', 'Relative_Humidity', 'Datetime_day', 'Datetime_month', 'Datetime_year', 'Datetime_hour'))
    prediction = run_inference(data)
    final_output = post_process(prediction)
    return final_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890)
