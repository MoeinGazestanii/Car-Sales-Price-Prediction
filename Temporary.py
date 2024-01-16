#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import streamlit as st

st.set_page_config(
    page_title="Used Cars Project",
    page_icon=":car:",
    layout="wide",
)
st.image('buying-a-used-car.jpg',width=500)


st.title('Used Cars Project')
st.write('The dataset contains information for predicting the selling price of used cars based on various factors such as mileage, engine power, number of seats, number of previous owners, and more. ')
st.write('The dataset was obtained from Kaggle. The URL for the dataset is https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv')
#Running the data

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Car_details.csv')
print(df)


# Basic Information
st.header('Dataset Information')
if st.button('Show Dataset'):
    st.dataframe(df)

print(df.columns)
print(df.info())
print(df.head(10))
# "Selling_Price" will be the dependent variable and the rest of the variables will be considered as independent variables.
st.write("Selling_Price will be the dependent variable and the rest of the variables will be considered as independent variables.")
st.sidebar.title('Variables table')
st.sidebar.write(pd.read_csv('C:\\Users\\Lenovo\\Desktop\\Car_details.csv').iloc[5:,[0,4]].reset_index(drop=True))

# Get the number of rows and columns
num_rows, num_columns = df.shape

# Print the number of rows and columns
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
print(f"Number of rows: {num_rows}")

print(f"Number of columns: {num_columns}")
#The dataset contains 8128 samples and 13 columns.

#Number of cars in Dataset
num_name = df['name'].value_counts()
print(num_name)

st.subheader('Cars Proportion')
show_num_cars = st.checkbox('Number of Cars')
# Display the number of cars if the checkbox is checked
if show_num_cars:
    num_name = df['name'].value_counts()
    st.write(num_name)
st.write('A large proportion of cars are Maruti')
# A large proportion of cars are Maruti

# Splitting the 'name' column into 'brand' and 'model' columns
df_copyname= df.copy()
df_copyname[['brand', 'model']] = df['name'].str.split(' ', n=1, expand=True)
print(df_copyname)

print(df_copyname['brand'].value_counts())

dfbrand = df_copyname['brand'].value_counts().sort_values()
first_10_items = dfbrand.head(20).index

df_copyname['brand'] = df_copyname['brand'].apply(lambda x: 'other' if x in first_10_items else x)

dfbrand_updated = df_copyname['brand'].value_counts().sort_values()

# Create a checkbox to show/hide the pie chart
show_pie_chart = st.checkbox('Show Pie Chart')
if show_pie_chart:
    fig=plt.figure(figsize=(10,6))
    plt.pie(dfbrand_updated, labels=dfbrand_updated.index, radius=1.3, autopct='%0.0f%%', shadow=True)
    st.write(fig)


st.subheader('Numerical stats')

#numerical stats
if st.button('Describe'):
    st.write(df.describe())
print(df.describe())

#missing values
print(df.isna().sum())


st.subheader("Missing Values")
def PercentageofMissingData(dataset):
    return dataset.isna().sum()/len(dataset)*100
if st.checkbox("Percentage of Missing values"):
    st.write(PercentageofMissingData(df))
st.text('At most two percent of a variable contains missing values')


df.dropna(inplace=True, axis=0)
st.subheader("Dublicated rows")
# Dublicated rows
df[df.duplicated()] #1189 dublicated rows
df.drop_duplicates(inplace=True)
st.write("Number of duplicated rows:", len(df.duplicated()))
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
df_copy['owner'].replace({'Test Drive Car':0, 'First Owner':1, 'Second Owner':2, 'Third Owner':3, 'Fourth & Above Owner':4}, inplace=True)
# Changing type of owner
df_copy['owner'] = pd.to_numeric(df_copy['owner'])

df_copy1 = pd.get_dummies(df_copy[['fuel', 'transmission', 'seller_type']],drop_first=True)
df_copy= pd.concat([df_copy, df_copy1], axis=1)
print(df_copy.head())

# Drop original columns
df_copy.drop(['fuel', 'transmission', 'seller_type'],axis=1, inplace=True)

#drop some unuseful columns in our model
df_copy = df_copy.drop(columns=['torque','name'],axis=1)
st.header("Dataset Preprocessing")
st.write('The data is preprocessed by encoding the target variable into numerical values and converting categorical variables into dummy variables.')
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
st.subheader("Outlier Detection")
outliers_km_driven = len(detect_outliers(df['km_driven']))
print("Number of Outliers detected in 'km_driven' column using IQR method:", outliers_km_driven)
st.write("Number of Outliers detected in 'km_driven' column using IQR method:",outliers_km_driven)
outliers_mileage = len(detect_outliers(df['mileage']))
print("Number of Outliers detected in 'mileage' column using IQR method:", outliers_mileage)
st.write("Number of Outliers detected in 'mileage' column using IQR method:",outliers_mileage)

outliers_engine = len(detect_outliers(df['engine']))
print("Number of Outliers detected in 'engine' column using IQR method:", outliers_engine)
st.write("Number of Outliers detected in 'engine' column using IQR method:",outliers_engine)

outliers_max_power = len(detect_outliers(df['max_power']))
print("Number of Outliers detected in 'max_power' column using IQR method:", outliers_max_power)
st.write("Number of Outliers detected in 'max_power' column using IQR method:",outliers_max_power)
st.header("Data Exploration")
with st.expander("Detail"):
    st.write('The target variable and features are plotted to explore data distribution and correlation.')
#correlation
correlation = df_copy.corr()
print(correlation)
plt.figure(figsize=(8,6))
sb.heatmap(correlation, annot=True)
st.subheader("Correlation")
Heatmap = st.checkbox("Heatmap")
if Heatmap:
    fig1 = plt.figure(figsize=(10, 6))
    sb.heatmap(correlation, annot=True)
    st.write(fig1)

with st.expander("Outcome"):
    st.write('- Diesel and petrol are the most common type of fuels in this dataset and it is the reason of a strong correlation between the dummy variables for these fuel types.')
    st.write('- There is a correlation between engine power and mileage. However, it is a negative correlation.')
    st.write('- Obviously, there is a negative correlation between the number of owners and year. The newer the car is, the fewer owners it tends to have.')
    st.write('- There is a strong positive correlation between the maximum  power of the car and our dependent variable selling price. so it is probably very effective in our model.')
    st.write('- By correlation, year and engine would be other important factors in selling price.')
# Diesel and petrol are the most common type of fuels in this dataset and it is the reason of a strong correlation between the dummy variables for these fuel types.
#There is a correlation between engine power and mileage. However, it is a negative correlation.
#Obviously, there is a negative correlation between the number of owners and year. The newer the car is, the fewer owners it tends to have.
#There is a strong positive correlation between the max power of the car and our dependent variable selling price. so it is probably very effective in our model.
# By correlation, year and engine would be other important factors in selling price.

# correlation of selling price with max power & engine plot
plt.figure(figsize=(10,6))
plt.scatter(df_copy['max_power'], df_copy['selling_price'],label="Max Power")
plt.scatter(df_copy['engine'], df_copy['selling_price'],label="Engine")
plt.xlabel("max power & engine")
plt.ylabel("selling price")
plt.title("correlation of selling price with max power & engine")
plt.legend()
plt.show()

# correlation of engine and mileage plot
plt.figure(figsize=(10, 6))
plt.scatter(df_copy['engine'], df_copy['mileage'])
plt.xlabel("engine")
plt.ylabel("mileage")
plt.title("correlation of engine and mileage")
plt.show()

# Selectbox to choose the plot
selected_plot = st.selectbox("Select Plot", ["Correlation with Max Power & Engine", "Correlation of Engine and Mileage","Selling Price by Year"])

# Plot based on the selection
if selected_plot == "Correlation with Max Power & Engine":
    if st.button("Show Correlation Plot"):
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df_copy['max_power'], df_copy['selling_price'], label="Max Power")
        plt.scatter(df_copy['engine'], df_copy['selling_price'], label="Engine")
        plt.xlabel("Max Power & Engine")
        plt.ylabel("Selling Price")
        plt.title("Correlation of Selling Price with Max Power & Engine")
        plt.legend()
        st.pyplot(fig)

elif selected_plot == "Correlation of Engine and Mileage":
    if st.button("Show Correlation Plot"):
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(df_copy['engine'], df_copy['mileage'])
        plt.xlabel("Engine")
        plt.ylabel("Mileage")
        plt.title("Correlation of Engine and Mileage")
        st.pyplot(fig)
elif selected_plot == "Selling Price by Year":
    if st.button("Show Selling Price by Year Plot"):
        fig = plt.figure(figsize=(20,10))
        plt.title('Selling price of cars by year')
        price_order = df_copy.groupby('year')['selling_price'].mean().sort_values(ascending=False).index.values
        sb.boxplot(data=df_copy, x='year', y='selling_price', order=price_order)
        st.pyplot(fig)

# Selling price of cars by year
fig = plt.figure(figsize=(10, 20))

plt.title('Selling price of cars by year')

price_order = df.groupby('year')['selling_price'].mean().sort_values(ascending=False).index.values

sb.boxplot(data=df, x='year', y='selling_price', order=price_order)


st.subheader("Visualization of int and float columns")
# Visualization of int and float columns

numeric_columns = []
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        numeric_columns.append(col)


for column in numeric_columns:
    plt.figure(figsize=(14, 6))

    # Plot histogram in the first subplot
    plt.subplot(1, 2, 1)
    plt.hist(df[column], rwidth=0.5, bins=10, color='skyblue', alpha=0.7)
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

selected_column = st.selectbox("Select Numeric Column", df.select_dtypes(['float64', 'int64']).columns)

# Button to trigger the visualization
if st.button(f"Visualize {selected_column}"):
    # Create a figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot histogram in the first subplot
    axes[0].hist(df[selected_column], rwidth=0.5, bins=10, color='skyblue', alpha=0.7)
    axes[0].set_title(f"Histogram of {selected_column}")
    axes[0].set_xlabel(selected_column)
    axes[0].set_ylabel("Frequency")

    # Plot horizontal box plot in the second subplot
    axes[1].boxplot(df[selected_column], vert=False)
    axes[1].set_title(f"Horizontal Boxplot of {selected_column}")
    axes[1].set_xlabel(selected_column)
    axes[1].set_ylabel("Values")
    # Show the plots
    st.pyplot(fig)

st.subheader("Categorical values visualization")
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

exclude_columns = ["name","torque"]
categorical_columns = [col for col in df.select_dtypes('object').columns if col not in exclude_columns]

# Selectbox to choose the categorical column
selected_column = st.selectbox("Select Categorical Column", categorical_columns)

# Button to trigger the visualization
if st.button(f"Visualize Frequency of {selected_column}"):
    # Plot frequency of the selected categorical column
    value_counts = df[selected_column].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(value_counts.index, value_counts.values, color='skyblue')
    ax.set_title(f'Frequency of {selected_column}')
    ax.set_xlabel(selected_column)
    ax.set_ylabel('Frequency')

    # Show the plots
    st.pyplot(fig)

#For an individual year, the sale price distribution looks like this:
fig = plt.figure(figsize=(10, 5))
plt.title("2015 sale price distributions")
sb.distplot(df[df['year']==2015].selling_price)

show_distribution_plot = st.checkbox('Show 2019 Sale Price Distribution Plot')

if show_distribution_plot:
    # Filter data for the year 2015
    data_2019 = df[df['year'] == 2019]

    # Create a distribution plot
    fig = plt.figure(figsize=(10, 5))
    plt.title("2019 Sale Price Distributions")
    sb.distplot(data_2019['selling_price'])
    # Show the distribution plot
    st.pyplot(fig)

# Overview of data distributions

fig = plt.figure(figsize=(10,6))
sb.pairplot(df.iloc[:,1:9])
st.write(fig)

# Train & Test
percent_to_use = 0.8
num_samples_to_use = int(len(df_copy) * percent_to_use)

x = df_copy.drop("selling_price", axis=1)
y = df_copy["selling_price"]

x_train, y_train = x[:num_samples_to_use], y[:num_samples_to_use]
x_test, y_test = x[num_samples_to_use:], y[num_samples_to_use:]

print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)
print('y_train shape',y_train.shape)
print('y_test shape',y_test.shape)

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import   mean_squared_error

reg= LinearRegression()
reg.fit(x_train,y_train)
reg.score(x_test,y_test)

# Coefficient
reg.coef_
# Intercept
reg.intercept_

# Make predictions on the training set
prediction=reg.predict(x_train)
print((prediction))
# Make predictions on the testing set
prediction_test= reg.predict(x_test)
print(prediction_test)

mean_squared_error(y_test,prediction_test)
np.sqrt(mean_squared_error(y_test,prediction_test))

# Creating Table
reg_summary = pd.DataFrame(x_train.columns, columns=["Features"])
reg_summary["Coefficients"] = reg.coef_
reg_summary["Intercept"] = reg.intercept_
print(reg_summary)

# Plot
plt.scatter(y_train, prediction, label='Training Data')
plt.scatter(y_test, prediction_test, label='Testing Data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], linestyle='--', color='black', label='Regression Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(x_train)
X_test_poly = poly.transform(x_test)

# Fit a linear regression model on the polynomial features
model_2 = LinearRegression()
model_2.fit(X_train_poly, y_train)

model_2.score(X_train_poly,y_train) #0.85

y_train_pred = model_2.predict(X_train_poly)

y_test_pred = model_2.predict(X_test_poly)

from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_test,y_test)

fig = plt.figure(figsize=(12,10))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
