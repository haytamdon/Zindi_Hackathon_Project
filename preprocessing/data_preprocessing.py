import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)

def date_processing(data):
    data = data.drop(["ID"], axis = 1)
    data['Datetime_day'] = data.Datetime.dt.day

    # month
    data['Datetime_month'] = data.Datetime.dt.month
    
    # year
    data['Datetime_year'] = data.Datetime.dt.year

    # hour
    data['Datetime_hour'] = data.Datetime.dt.hour

    return data

def missing_values_processing(data):
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(data.iloc[:,1:3])
    data.iloc[:,1:3]= imputer.transform(data.iloc[:,1:3])
    data["Temperature"] = data.groupby(["Datetime_month","Datetime_hour"])["Temperature"].apply(lambda x:x.fillna(x.median()))
    data["Temperature"] = data["Temperature"].apply(np.ceil)
    data["Relative_Humidity"] = data.groupby(["Datetime_month","Datetime_hour"])["Relative_Humidity"].apply(lambda x:x.fillna(x.median()))
    data["Relative_Humidity"] = data["Relative_Humidity"].apply(np.ceil)
    return data

def outlier_processing(data):
    scaler = StandardScaler()
    power = PowerTransformer(method='yeo-johnson')
    data.iloc[:,1:5] = scaler.fit_transform(data.iloc[:,1:5])
    data.iloc[:,1:5] = power.fit_transform(data.iloc[:,1:5])
    return data