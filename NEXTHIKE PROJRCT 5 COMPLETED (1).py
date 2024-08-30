#!/usr/bin/env python
# coding: utf-8

# # NEXTHIKE PROJECT 5 

# In[6]:


#Task 1: User Overview Analysis
#1.1: Load and Prepare Data


# In[7]:


import pandas as pd

# Load datasets
telecom_data = pd.read_csv('telcom_data.csv')
file_description = pd.read_csv('field_data.csv')



# In[8]:


telecom_data


# In[9]:


file_description


# In[10]:


# Display basic info
print(telecom_data.info())


# In[11]:


print(file_description.info())


# In[12]:


#1.2: Aggregate User Behavior Metrics
# Aggregate metrics per user
user_metrics = telecom_data.groupby('IMSI').agg({
    'Dur. (ms)': 'sum',
    'Avg RTT DL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum'
}).reset_index()

print(user_metrics.head())


# In[13]:


##Exploratory Data Analysis (EDA)
#Non-Graphical Univariate Analysis
# Summary statistics
stats = telecom_data[['Dur. (ms)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']].describe()
print(stats)


# In[14]:


##Graphical Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of 'Dur. (ms)'
sns.histplot(telecom_data['Dur. (ms)'])
plt.title('Distribution of Call Duration')
plt.xlabel('Duration (ms)')
plt.ylabel('Frequency')
plt.show()

# Similar plots for other variables


# In[15]:


##Bivariate Analysis

# Scatter plot of Avg Bearer TP DL vs. Total DL (Bytes)
sns.scatterplot(x='Avg Bearer TP DL (kbps)', y='Total DL (Bytes)', data=telecom_data)
plt.title('Average Bearer Throughput vs. Total Download Bytes')
plt.xlabel('Average Bearer TP DL (kbps)')
plt.ylabel('Total DL (Bytes)')
plt.show()


# In[16]:


#Variable Transformations
# Segment users based on 'Dur. (ms)'
user_metrics['DurationSegment'] = pd.cut(user_metrics['Dur. (ms)'], bins=[0, 60000, 120000, 180000, 240000], labels=['Short', 'Medium', 'Long', 'Very Long'])
print(user_metrics[['DurationSegment', 'Total DL (Bytes)']].groupby('DurationSegment').mean())


# In[17]:


##Correlation Analysis
# Compute and plot correlation matrix
corr_matrix = telecom_data[['Dur. (ms)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[18]:


##Dimensionality Reduction
from sklearn.decomposition import PCA

# Prepare data for PCA
features = telecom_data[['Dur. (ms)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']]
pca = PCA(n_components=2)

reduced_data = pca.fit_transform, ('X','features')
print(reduced_data[:5])


# In[19]:


#Task 2: User Engagement Analysis
#Aggregate and Normalize Metrics


# In[20]:


from sklearn.preprocessing import StandardScaler

# Aggregate engagement metrics
engagement_metrics = telecom_data.groupby('IMSI').agg({
    'Dur. (ms)': 'sum',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum'
})

# Normalize metrics
scaler = StandardScaler()
normalized_engagement = scaler.fit_transform(engagement_metrics)
print(normalized_engagement[:5])


# In[21]:


#K-Means Clustering
from sklearn.cluster import KMeans

# K-Means clustering
kmeans = KMeans(n_clusters=3)
engagement_metrics['Cluster'] = kmeans.fit_predict(normalized_engagement)
print(engagement_metrics.groupby('Cluster').mean())


# In[22]:


#Top Engaged Users per Application
# Top users by total download bytes
top_users = telecom_data[['IMSI', 'Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)','Total DL (Bytes)']]
top_users = top_users.groupby('IMSI').sum()
print(top_users)


# In[23]:


#Optimal Number of Clusters (Elbow Method)
# Elbow method for optimal k
inertia = []
for k in range(5, 9):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(normalized_engagement)
    inertia.append(kmeans.inertia_)

plt.plot(range(5, 9), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[24]:


#Task 3: Experience Analytics
#Aggregate Experience Metrics


# In[25]:


# Aggregate experience metrics
experience_metrics = telecom_data.groupby('IMSI').agg({
    'Avg RTT DL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Handset Manufacturer': 'first',  # Assuming one handset per user
    'Handset Type': 'first',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum'
}).reset_index()

print(experience_metrics.head())


# In[26]:


#Top/Bottom/Frequent Values
# Compute top, bottom, and frequent values for RTT
top_rtt = telecom_data['Avg RTT DL (ms)'].nlargest(10)
bottom_rtt = telecom_data['Avg RTT DL (ms)'].nsmallest(10)
frequent_rtt = telecom_data['Avg RTT DL (ms)'].mode()

print(top_rtt)
print(bottom_rtt)
print(frequent_rtt)


# In[27]:


#Distribution of Throughput per Handset Type
# Box plot of throughput by handset type
sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=experience_metrics)
plt.title('Throughput Distribution by Handset Type')
plt.show()


# In[28]:


#Remove Rows with Missing Values
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Assuming `experience_metrics` is your DataFrame
experience_features = experience_metrics[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']]

# Drop rows with missing values
experience_features_clean = experience_features.dropna()
experience_metrics_clean = experience_metrics.loc[experience_features_clean.index]

# Apply PCA
pca = PCA(n_components=2)
reduced_experience_data = pca.fit_transform(experience_features_clean)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
experience_metrics_clean['Cluster'] = kmeans.fit_predict(reduced_experience_data)

# Print mean of each cluster
print(experience_metrics_clean.groupby('Cluster').mean())



# In[29]:


#Impute Missing Values
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Assuming `experience_metrics` is your DataFrame
experience_features = experience_metrics[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']]

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
experience_features_imputed = imputer.fit_transform(experience_features)

# Apply PCA
pca = PCA(n_components=2)
reduced_experience_data = pca.fit_transform(experience_features_imputed)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
experience_metrics['Cluster'] = kmeans.fit_predict(reduced_experience_data)

# Print mean of each cluster
print(experience_metrics.groupby('Cluster').mean())


# In[30]:


#After imputing missing values, you can proceed with PCA and clustering:
pca = PCA(n_components=2)
reduced_experience_data = pca.fit_transform(experience_features_imputed)
kmeans = KMeans(n_clusters=3)
experience_metrics['Cluster'] = kmeans.fit_predict(reduced_experience_data)
print(experience_metrics.groupby('Cluster').mean())


# In[31]:


#Check for Missing Values
print(experience_features.isnull().sum())


# In[32]:


#Here's a complete example using imputation:
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Assuming `experience_metrics` is your DataFrame
experience_features = experience_metrics[['Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)', 'Total DL (Bytes)', 'Total UL (Bytes)']]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
experience_features_imputed = imputer.fit_transform(experience_features)

# Apply PCA
pca = PCA(n_components=2)
reduced_experience_data = pca.fit_transform(experience_features_imputed)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
experience_metrics['Cluster'] = kmeans.fit_predict(reduced_experience_data)

# Print mean of each cluster
print(experience_metrics.groupby('Cluster').mean())

