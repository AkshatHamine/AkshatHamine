#!/usr/bin/env python
# coding: utf-8

# # NEXTHIKE PROJECT 8

# In[ ]:


##Before starting, ensure we have the required libraries installed. we can install them using:


# In[1]:


pip install pandas numpy matplotlib seaborn scikit-learn tensorflow flask docker


# In[4]:


#Load the Data
#First, load the CSV data into a Pandas DataFrame:
import pandas as pd

# Load the dataset
df = pd.read_csv('all_upwork_jobs_P8.csv')


# In[6]:


df


# In[7]:


# Display the first few rows of the dataset
print(df.info())


# In[8]:


# Display the first few rows of the dataset
print(df.head())


# In[9]:


# Display the first few rows of the dataset
print(df.tail())


# In[ ]:


##Task 1: Analyze the Correlation Between Job Title Keywords and Offered


# In[15]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('all_upwork_jobs_P8.csv')

# Fill missing values
data['hourly_low'].fillna(0, inplace=True)
data['hourly_high'].fillna(0, inplace=True)



# In[16]:


data


# In[17]:


# Create a salary range column
data['average_hourly'] = data[['hourly_low', 'hourly_high']].mean(axis=1)



# In[18]:


data


# In[31]:


# Handle missing values
data['title'] = data['title'].fillna('')
data['hourly_low'].fillna(0, inplace=True)
data['hourly_high'].fillna(0, inplace=True)


# In[32]:


data


# In[33]:


# Create a salary range column
data['average_hourly'] = data[['hourly_low', 'hourly_high']].mean(axis=1)


# In[34]:


data


# In[40]:


# Feature extraction from job titles
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['title'])
y = data['average_hourly']

# Linear regression model
model = LinearRegression()
model.fit(X, y)


# In[41]:


X


# In[47]:


y


# In[45]:


# Coefficients of the keywords
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

# Create a DataFrame of keywords and their coefficients
coef_df = pd.DataFrame({'keyword': feature_names, 'coefficient': coefficients})
coef_df = coef_df.sort_values(by='coefficient', ascending=False)



# In[46]:


coef_df


# In[44]:


# Plot the top 20 keywords
top_keywords = coef_df.head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x='coefficient', y='keyword', data=top_keywords)
plt.title('Top 20 Keywords Correlated with Average Hourly Rate')
plt.show()


# In[48]:


#Task 2: Identify Emerging Job Categories Based on Posting Frequency
# Extract job categories (assuming 'title' can be used to infer categories)
data['category'] = data['title'].str.split().str[0]  # Example: Using the first word as a category

# Frequency of job categories
category_counts = data['category'].value_counts()

# Plot the top 20 categories
top_categories = category_counts.head(20)
plt.figure(figsize=(12, 8))
sns.barplot(x=top_categories.index, y=top_categories.values)
plt.title('Top 20 Job Categories by Posting Frequency')
plt.xticks(rotation=90)
plt.show()


# In[52]:


##Task 3: Predict High-Demand Job Roles by Analyzing Job Posting Patterns
# Convert 'published_date' to datetime
data['published_date'] = pd.to_datetime(data['published_date'])


# In[53]:


data


# In[54]:


# Extract year and month
data['year_month'] = data['published_date'].dt.to_period('M')


# In[55]:


data


# In[56]:


# Count job postings per month
monthly_counts = data.groupby(['year_month', 'category']).size().reset_index(name='count')


# In[57]:


monthly_counts


# In[58]:


# Plot trends for the top 5 categories
top_categories = monthly_counts['category'].value_counts().head(5).index
top_categories_data = monthly_counts[monthly_counts['category'].isin(top_categories)]

plt.figure(figsize=(14, 10))
sns.lineplot(data=top_categories_data, x='year_month', y='count', hue='category')
plt.title('Monthly Job Postings Trends for Top 5 Categories')
plt.xticks(rotation=45)
plt.show()


# In[59]:


# Convert 'published_date' to datetime
data['published_date'] = pd.to_datetime(data['published_date'])

# Extract year and month
data['year_month'] = data['published_date'].dt.to_period('M')

# Count job postings per month
monthly_counts = data.groupby(['year_month', 'category']).size().reset_index(name='count')


# In[27]:


#Task 4: Compare Average Hourly Rates Across Different Countries
#To compare hourly rates by country, you can aggregate and visualize the data.


# In[28]:


# Average hourly rate by country
country_rates = data.groupby('country')['average_hourly'].mean().reset_index()

# Plot the average hourly rates by country
plt.figure(figsize=(14, 8))
sns.barplot(x='country', y='average_hourly', data=country_rates)
plt.title('Average Hourly Rates by Country')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




