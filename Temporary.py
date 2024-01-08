#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb



#Running the data

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Car_details.csv')
print(df)


# Basic Information

print(df.columns)
print(df.info())
print(df.head(10))
# "Selling_Price" will be the dependent variable and the rest of the variables will be considered as independent variables.

# Get the number of rows and columns
num_rows, num_columns = df.shape

# Print the number of rows and columns
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
#The dataset contains 8128 samples and 13 columns.


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

df_copy= df.copy()

# Encoding Fuel type
df_copy['fuel'].replace({"CNG":"CNG/LPG", "LPG": "CNG/LPG"}, inplace= True)

# Encoding Owner
df_copy['owner'].replace({'Test Drive Car':0, 'First Owner':1, 'Second Owner':2, 'Third Owner':3, 'Fourth & Above Owner':3}, inplace=True)
df_copy['owner'] = pd.to_numeric(df_copy['owner'])

df_copy1 = pd.get_dummies(df_copy[['fuel', 'transmission', 'seller_type']],drop_first=True)
df_copy= pd.concat([df_copy, df_copy1], axis=1)
print(df_copy.head())
df_copy.drop(['fuel', 'transmission', 'seller_type'],axis=1, inplace=True)

#drop some columns
df_copy=df_copy.drop(columns=['torque','name'],axis=1)


#outlier Detection
def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_indices = []
    for val in data:
        if val < lower_bound or val > upper_bound:
            outliers_indices.append(val)
    return outliers_indices

outliers_km_driven = detect_outliers(df['km_driven'])
print("Outliers detected in 'km_driven' column using IQR method:", outliers_km_driven)

outliers_mileage = detect_outliers(df['mileage'])
print("Outliers detected in 'mileage' column using IQR method:", outliers_mileage)

outliers_engine = detect_outliers(df['engine'])
print("Outliers detected in 'engine' column using IQR method:", outliers_engine)

outliers_max_power = detect_outliers(df['max_power'])
print("Outliers detected in 'max_power' column using IQR method:", outliers_max_power)

#correlation
correlation = df_copy.corr()
print(correlation)
plt.figure(figsize=(8,6))
sb.heatmap(correlation, annot=True)

#Basic Visualization of int and float columns

numeric_columns = []
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        numeric_columns.append(col)


for column in numeric_columns:
    plt.figure(figsize=(14, 6))

    # Plot histogram in the first subplot
    plt.subplot(1, 2, 1)
    plt.hist(df[column], rwidth=0.5, bins=20, color='blue', alpha=0.7)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # Plot horizontal box plot in the second subplot
    plt.subplot(1, 2, 2)
    plt.boxplot(df[column], vert=False)
    plt.title(f"Horizontal Boxplot of {column}")
    plt.xlabel(column)
    plt.ylabel("Values")
    plt.show()


    # Categorical values visualization
    plt.figure(figsize=(8, 6))
    plt.subplot(2,2,1)
    value_counts = df['transmission'].value_counts()
    plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.title('Frequency of Transmission Types')
    plt.xlabel('Transmission Type')
    plt.ylabel('Frequency')
    plt.subplot(2,2,2)
    value_counts2 = df['seller_type'].value_counts()
    plt.bar(value_counts2.index, value_counts2.values, color='skyblue')
    plt.title('Frequency of Seller Types')
    plt.xlabel('Seller Type')
    plt.ylabel('Frequency')
    plt.subplot(2,2,3)
    value_counts3=df['fuel'].value_counts()
    plt.bar(value_counts3.index,value_counts3.values,color='skyblue')
    plt.title('Frequency of Fuel Types')
    plt.xlabel('Fuel Type')
    plt.ylabel('Frequency')
    plt.subplot(2,2,4)
    value_counts4 = df['owner'].value_counts()
    plt.bar(value_counts4.index, value_counts4.values, color='skyblue')
    plt.title('Frequency of Owners')
    plt.xlabel('Owner')
    plt.ylabel('Frequency')
    plt.show()




















