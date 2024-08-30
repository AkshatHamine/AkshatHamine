#!/usr/bin/env python
# coding: utf-8

# # NEXTHIKE PROJECT 6

# In[1]:


##Setup and Data Loading 
import pandas as pd
# Load the data
train_data = pd.read_csv('train_p8.csv')
test_data = pd.read_csv('test._p8csv.csv')
store_data = pd.read_csv('store_p8.csv')
submission_data = pd.read_csv('sample_submission_p8.csv')


# In[2]:


train_data


# In[3]:


test_data


# In[4]:


store_data


# In[5]:


submission_data


# In[6]:


# Display the first few rows of each dataset
print(train_data.head())


# In[7]:


print(test_data.head())


# In[8]:


print(store_data.head())


# In[9]:


print(submission_data.head())


# In[10]:


##Exploratory Data Analysis (EDA)
#Data Cleaning
# Check for missing values
print(train_data.isnull().sum())


# In[11]:


print(test_data.isnull().sum())


# In[12]:


print(store_data.isnull().sum())


# In[13]:


print(submission_data.isnull().sum())


# In[14]:


# Handle missing values (Example: Fill NaNs with the median or use a strategy of your choice)
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)


# In[15]:


train_data
test_data


# In[16]:


# Check for duplicates
print(train_data.duplicated().sum())
print(test_data.duplicated().sum())


# In[17]:


##Visualizing Features
import matplotlib.pyplot as plt
import seaborn as sns

# Sales distribution
sns.histplot(train_data['Sales'], bins=50, kde=True)
plt.title('Sales Distribution')
plt.show()


# In[18]:


# Sales by Store Type
sns.boxplot(data=train_data, x='Store', y='Sales')
plt.title('Sales by Store')
plt.show()


# In[19]:


# Promo impact
sns.boxplot(data=train_data, x='Promo', y='Sales')
plt.title('Sales During and Outside Promo')
plt.show()


# In[20]:


##Correlation and Seasonality
# Correlation between Sales and Number of Customers
sns.scatterplot(data=train_data, x='Customers', y='Sales')
plt.title('Sales vs Number of Customers')
plt.show()


# In[21]:


# Sales Before, During, and After Holidays
train_data['StateHoliday'] = train_data['StateHoliday'].map({'0': 'None', 'a': 'Public Holiday', 'b': 'Easter Holiday', 'c': 'Christmas'})
sns.boxplot(data=train_data, x='StateHoliday', y='Sales')
plt.title('Sales Behavior During Holidays')
plt.show()


# In[22]:


##Feature Engineering
from sklearn.preprocessing import StandardScaler
import numpy as np

# Feature Engineering
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
test_data['DayOfWeek'] = test_data['Date'].dt.dayofweek


# In[23]:


train_data
test_data


# In[24]:


# Additional features
train_data['IsWeekend'] = train_data['DayOfWeek'] >= 5
test_data['IsWeekend'] = test_data['DayOfWeek'] >= 5


# In[25]:


train_data
test_data


# In[26]:


# Scaling features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_data[['Sales', 'Customers']])
train_data[['Sales', 'Customers']] = scaled_features


# In[27]:


train_data


# In[28]:


##Machine Learning Modeling
#Random Forest Regressor with Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[ ]:


# Define features and target
X_train = train_data.drop(['Sales', 'Date'], axis=1)
y_train = train_data['Sales']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), ['Customers']),
        ('cat', OneHotEncoder(), ['Store', 'Promo', 'StateHoliday'])
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_train)
print(f'Mean Squared Error: {mean_squared_error(y_train, y_pred)}')


# In[ ]:


X_train
y_train 


# In[ ]:


##Deep Learning with LSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data for LSTM
def create_lstm_dataset(data, target_column, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        X.append(data[i:end_ix])
        y.append(data[end_ix])
    return np.array(X), np.array(y)

# Transform data
n_steps = 10
X, y = create_lstm_dataset(train_data['Sales'].values, 'Sales', n_steps)

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=1)

# Predict using the model
X_test, y_test = create_lstm_dataset(test_data['Sales'].values, 'Sales', n_steps)
y_pred = model.predict(X_test)


# In[ ]:


X_test 
y_test

