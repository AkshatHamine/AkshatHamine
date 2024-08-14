#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# NEXT HIKE PROJECTS 2...

#Task 1...

Data Acuisition and data Warngling on dataset 1 and 2
# In[2]:


#now do import pandas and Numpy libraries

import pandas as pd
import numpy as np


# In[3]:


#now do import Database1
database1=pd.read_csv("Database1.csv")


# In[4]:


#show database1
database1


# In[5]:


#now do import database2
database2=pd.read_csv("database2.csv")


# In[6]:


#show database2
database2


# In[33]:


#now identify the Datatype of database1
database1.info


# In[34]:


#now identify the Datatype database2
database2.info


# In[7]:


#now indentify some Top5 data of database1
values database1.head()


# In[8]:


#now identify the Top5 data  of database2
database2.head


# In[12]:


#now identify Bottom 5 data of database1 
database1.tail()


# In[13]:


#now identify the Bottom 5 data of database2
database2.tail()


# In[15]:


#now indentify row & columns of database1 
database1.shape


# In[16]:


#now identify row & columns of database2
database2.shape


# In[17]:


#now identify only rows of database1
database1.shape[0]


# In[22]:


#now indentify only column of database1
database1.shape[1]


# In[23]:


#now identify only row of database2
database2.shape[0]


# In[36]:


#now indentify only column of database2
database2.shape[1]


# In[37]:


#now we identify name of columns if database1
database1.columns


# In[38]:


#now we identify name of columns in databse2
database2.columns


# In[39]:


#Now we combine two database:- Database1 &d Database2
databaseMrg=pd.merge(database1,database2,on='instant',how='inner')


# In[41]:


#now we got the merged database in "databaseMrg"
databaseMrg


# In[43]:


#now we identify the unique value in databaseMrg
databaseMrg['weathersit'].unique()


# In[45]:


databaseMrg['temp'].unique()


# In[47]:


databaseMrg['dteday'].unique()


# In[48]:


databaseMrg['yr'].unique()


# In[49]:


databaseMrg['mnth'].unique()


# In[50]:


databaseMrg['season'].unique()


# In[51]:


databaseMrg['weekday'].unique()


# In[60]:


#now we remove the unnecessary columns
database_frs = databaseMrg.drop(['Unnamed: 0','yr','mnth'], axis = 1)


# In[62]:


#now we got the fresh databse by removing the unnecessary columns
database_frs


# In[64]:


#now we identify the missing values
database_frs.isnull().sum()


# In[65]:


#now we identify rows & colunms 
database_frs.shape


# In[67]:


#now we identify the daytatype of database_frs
database_frs.info()


# In[70]:


#now we identify the datatype of database_frs with method
database_frs.dtypes


# In[71]:


#now we indentify the summery of database_frs 
database_frs.describe()


# In[72]:


#Treat missing value
database_frs


# In[81]:


#now we identify which data who have only zero value
dfc= database_frs.ffill()


# In[82]:


#now we got data who have only zero value
dfc.isnull().sum()


# In[83]:


#now we change 1 with the word spring in season in dfc
dfc['season'].replace({1:'spring'},inplace =True)


# In[84]:


dfc


# In[88]:


#now we put some information for more classify the data after replacing method
dfc['weekday'].replace({0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'},inplace=True)


# In[89]:


dfc


# In[92]:


#now we got data who have only zero value
dfc.isnull().sum()


# In[93]:


#now we check the datatype of dfc
dfc.info()


# # Task 2
DataAcquisition and Wrangling on Database 3
# In[98]:


#now do import Database3
database3=pd.read_csv("database3.csv")


# In[99]:


database3


# In[101]:


#now we drop unnecessary columns from database3
db3 = database3.drop(['yr','mnth'],axis =1)


# In[102]:


db3


# In[104]:


#now we indetify the Bottom 9 value of data
tail = db3.tail(9)


# In[105]:


tail


# In[106]:


#now ewe drop some value form data
db4 = db3.drop([381,382,383,384,385,386,387,388,389],axis = 0)


# In[107]:


db4


# In[109]:


#now we sort the value of data
db4 = db4.sort_values('instant')


# In[110]:


db4


# In[111]:


#now we reset data
db4 = db4.reset_index()


# In[112]:


db4


# In[113]:


#now we drop unnecessary column
db4 = db4.drop('index', axis = 1)


# In[114]:


db4

## now we concatenate db4 with this dataset 3
# In[118]:


concatenated_data = pd.concat((dfc,db4),axis = 0)


# In[119]:


concatenated_data


# In[124]:


#now we drop unnecessary column
db4 = db3.drop('hr' , axis = 1)


# In[125]:


db4


# In[126]:


#now we put some information (season) for more classify about the data after replacing method
concatenated_data['season'].replace({1:'spring'},inplace =True)


# In[127]:


concatenated_data


# In[128]:


#now we check the mising values
concatenated_data.isnull().sum()


# In[130]:


#now we check the duplicate value in data
duplicate_rows = concatenated_data[concatenated_data.duplicated()]

#now we ewmove duplicates
concatenated_data.drop_duplicates()


# In[135]:


#now we find outliers for a specific columns

Q1 = concatenated_data['cnt'].quantile(0.25)
Q3 = concatenated_data['cnt'].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

#Remove Qutliers

df = concatenated_data[(concatenated_data['cnt']>=lower_fence) & (concatenated_data['cnt']<=upper_fence)]


# In[136]:


df


# In[137]:


df1 = df.reset_index()


# In[138]:


df1

##Check the skewness and correletion of the data
# In[140]:


# now we we to do first import the matplotlib and seaborn liabraries

import matplotlib.pyplot as plt
import seaborn as sns


# In[142]:


skewness = df1['cnt'].skew()
print(f"skewness: {skewness}")

#now we've to make a histogram graph to see the distribution

plt.hist(df1['cnt'],bins=20)  #we can adjust the enter of bins
plt.xlabel('values')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.show()


# In[145]:


#now we doCorrelation of the data
df1.corr()


# In[ ]:





# In[ ]:




