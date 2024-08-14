#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##NEXT HIKE PROJECTS 1 


# In[1]:


#Import the libraries
import requests
import string


# In[4]:


#now we importt Beautiful library 
from bs4 import BeautifulSoup 
#which is use for webscraping


# In[5]:


#now write code for fetch the 
def get_latest_python_articles():
    url = "https://www.python.org/"
    response=requests.get(url)
    
    if response.status_code== 200:
        soup=BeautifulSoup(response.text,"html.parser")
        latest_articles=[]
        
        for article in soup.select(".blog-datawidget li"):
            title=article.a.text.strip()
            latest_articles.append(title)
        
        return latest_articles
    else:
        print(f"failed to retrieve data. status code: {response.status_code}")
        return[]
if __name__=="__main__":
    python_articles = get_latest_python_articles()
        
    if python_articles:
        print("latest article in python section:")
        for index,article in enumerate(python_articles,1):
            print(f"{index}.{article}")
        else:
            print("No article found.")


# In[48]:


#import the immportant libraries
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from collections import Counter


# In[39]:


#now use URl
url="https://www.python.org/"


# In[40]:


# now this is use to takeout html code from website or URl
html = requests.get(url)


# In[41]:


#now we comvert html code in text file
html_doc = html.text


# In[42]:


#now we use this to structured all the tags in tree formate
soup = BeautifulSoup(html_doc, 'html.parser')


# In[43]:


soup.prettify()


# In[44]:


#now we use to show the text
print(soup.get_text())


# In[49]:


#now we use this to show only content
word = []
for paragraph in soup.find_all('p'):
    paral = paragraph.text
    cleaned_text = re.sub('A-Za-z\s' , '', paral)
    print(cleaned_text)


# In[50]:


#now we use this code to count data
def counting():
    
    all_words = []
    filename2 = 'counting.txt'
    
    for paragraph in soup.find_all('p'):
        paral = paragraph.text
    cleaned_text = re.sub(r'[A-Za-z\s]','', paral)
    words = paral.split()
    all_words.extend(words)
            
    word_count = Counter(all_words)
        
    for word,count in word_count.items():
        print(f"'{word}' occurs {count} times")
            
    words_string = ' '.join(all_words)
        
    with open(filename2, 'a') as file2:
        file2.write(words_string)
        
counting()

