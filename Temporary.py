#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb



#Running the data

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Car_details.csv')
print(df)
Fun.PercentageofMissingData(df)
# Basic Information
print(df.shape)     #The dataset contains 8128 samples and 13 columns.

print(df.columns)
print(df.info())
print(df.head(10))
# "Selling_Price" will be the dependent variable and the rest of the variables will be considered as independent variables.




#numerical stats
print(df.describe())

#missing values
print(df.isna().sum())

def PercentageofMissingData(dataset):
    return dataset.isna().sum()/len(dataset)*100

PercentageofMissingData(df)
# At most two percent of a variable contains missing values

df.dropna(inplace=True, axis=0)

df[df.duplicated()] #1189 dublicated rows
df.drop_duplicates(inplace=True)

#Data Preprocessing
    def change(data, columns, string_to_replace, replacement):
        for col in columns:
            data[col] = data[col].replace(string_to_replace, replacement,regex=True)
        return data

change(df, ['mileage'], ' kmpl', '')
change(df, ['engine'], ' CC', '')
change(df, ['max_power'], ' bhp', '')
change(df, ['mileage'], ' km/kg', '')

print(df.info())
# changing the type of some variables
df['mileage'] = pd.to_numeric(df['mileage'])
df['engine'] = pd.to_numeric(df['engine'])
df['max_power'] = pd.to_numeric(df['max_power'])
print(df.info())

# Encoding seller type
df_copy= df.copy()
df_copy['seller_type'].replace({"Individual":0, "Dealer":1, "Trustmark Dealer": 2}, inplace= True)
# Encoding Fuel
df_copy['fuel'].replace({"CNG":0, "Diesel":1, "LPG": 2, "petrol":3}, inplace= True)












