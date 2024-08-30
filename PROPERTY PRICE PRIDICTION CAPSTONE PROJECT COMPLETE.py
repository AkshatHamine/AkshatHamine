#!/usr/bin/env python
# coding: utf-8

# # PROPERTY PRICE PRIDICTION CAPSTONE PROJECT...

# In[7]:


#IMMPORT LIBRARIES AND LOAD DATA
import pandas as pd

# Load the dataset
file_path = 'Property_data (1).csv'
data = pd.read_csv(file_path)


# In[8]:


data


# In[9]:


# Display the first few rows of the dataset
print(data.head())


# In[10]:


print(data.info())


# In[69]:


##Data Cleaning


# In[70]:


# Fill missing values for numerical columns with median
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].median(), inplace=True)

# Fill missing values for categorical columns with the most frequent value
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Check for any remaining missing values
print(data.isnull().sum())


# In[71]:


##Exploratory Data Analysis (EDA)
# Display summary statistics
print(data.describe())


# In[19]:


# Fill missing values for categorical columns with the most frequent value
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Check for any remaining missing values
print(data.isnull().sum())


# In[28]:


##Drop Unnecessary Columns:
# Drop columns that are not needed (e.g., 'Property ID')
# Note: Adjust column names as per your actual dataset
if 'Property ID' in data.columns:
    data.drop(['Property ID'], axis=1, inplace=True)


# In[29]:


data


# In[32]:


##Correlation Matrix:
import seaborn as sns
import matplotlib.pyplot as plt

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[34]:


##Distribution of the Target Variable:
# Assuming 'SalePrice' is the target variable
plt.figure(figsize=(10, 6))
sns.histplot(data['PropPrice'], kde=True)
plt.title('Sale Price Distribution')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()


# In[35]:


##Feature Engineering
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


# In[36]:


le


# In[37]:


data


# In[54]:


##Create New Features:
# Example feature: Age of the property
data['PropertyClass'] = data['SaleYr'] - data['PropPrice']


# In[55]:


data


# In[75]:


##Feature Selection and Scaling:
from sklearn.preprocessing import StandardScaler

# Select features for model
features = ['PropertyClass', 'PropertySize', 'PropertyFrontage', 'PropPrice']  # Update as needed
X = data[features]
y = data['PropPrice']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[80]:


X_scaled


# In[81]:


##Model Training and Evaluation
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[86]:


X_train, y_train


# In[87]:


X_test, y_test


# In[91]:


train_test_split


# In[82]:


##Train a Machine Learning Model:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[94]:


##Evaluate the Model:
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')


# In[98]:


# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')


# In[99]:


##Reporting and Saving Results
import joblib

# Save the trained model
joblib.dump(model, 'property_price_model.pkl')

# Save predictions for future use
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('model_predictions.csv', index=False)


# In[101]:


results

