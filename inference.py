import json
import requests
import pandas as pd
from preprocessing.data_preprocessing import date_processing


def test_processing(data):
    X_test = data.iloc[:,1:]
    X_test.dropna(inplace= True)
    return X_test

def request_inference(data):
    data = date_processing(data)
    data = test_processing(data)
    for i in range(len(data)):
        f = {'Sensor1_PM2.5': str(data['Sensor1_PM2.5'].iloc[i]),
            'Sensor2_PM2.5': str(data['Sensor2_PM2.5'].iloc[i]),
            'Temperature': str(data['Temperature'].iloc[i]),
            'Relative_Humidity': str(data['Relative_Humidity'].iloc[i]),
            'Datetime_day' : str(data['Datetime_day'].iloc[i]),
            'Datetime_month' : str(data['Datetime_month'].iloc[i]),
            'Datetime_year' : str(data['Datetime_year'].iloc[i]),
            'Datetime_hour': str(data['Datetime_hour'].iloc[i])}
        r = requests.post('http://localhost:8890/test', files=f)
        response = r.text
        print(response)

if __name__ == "__main__":
    inference = pd.read_csv('C:/Users/XX/Documents/Data Science/Projects/umojahack-africa-2022-beginner-challenge/test.csv', parse_dates=['Datetime'])
    request_inference(inference)