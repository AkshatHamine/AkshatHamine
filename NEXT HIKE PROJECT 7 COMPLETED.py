#!/usr/bin/env python
# coding: utf-8

# # NEXTHIKE PROJECT 7

# In[3]:


##Part1 Data Exploration and Preparation
import pandas as pd

# Load the dataset
df = pd.read_csv('twitter_disaster.csv')
df


# In[5]:


##Explore the Dataset
#Inspect the structure: Check columns, data types, and for any missing values.
print(df.info())


# In[6]:


print(df.head())


# In[7]:


##Visualize Class Distribution
#How: Use histograms or bar plots to visualize the number of disaster vs. non-disaster tweets.
import matplotlib.pyplot as plt

df['target'].value_counts().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Number of Tweets')
plt.title('Distribution of Disaster and Non-Disaster Tweets')
plt.show()


# In[13]:


pip install wordcloud


# In[14]:


##Analyze Keywords and Phrases
#How: Identify and count common keywords or phrases in disaster-related tweets.
from wordcloud import WordCloud


# In[15]:


from wordcloud import WordCloud

disaster_texts = df[df['target'] == 1]['text']
disaster_words = ' '.join(disaster_texts)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(disaster_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[17]:


##Data Preparation
#Clean the Text Data
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    return text.lower()

df['cleaned_text'] = df['text'].apply(clean_text)


# In[18]:


df


# In[19]:


##Tokenize the Text
#How: Split the text into individual words or tokens.
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])


# In[20]:


X


# In[21]:


##Convert Text Labels to Numerical Format
#How: Encode the target variable (e.g., 0 for non-disaster, 1 for disaster).

df['target'] = df['target'].astype(int)


# In[23]:


df


# In[41]:


##Split the Dataset
#How: Divide the data into training and testing sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.2, random_state=42)


# In[42]:


y_train


# In[43]:


y_test


# In[ ]:


##Part 2:- Feature Engineering and Model Selection
#Task: Feature Engineering


# In[44]:


##Extract Features
#How: Use methods like TF-IDF to transform text data into numerical features./
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])


# In[45]:


X


# In[47]:


pip install gensim nltk


# In[52]:


from gensim.models import Word2Vec
import gensim


# In[53]:


##Use Pre-trained Word Embeddings (Optional)
#How: Implement embeddings like Word2Vec or GloVe to capture semantic meaning.
from gensim.models import Word2Vec


# In[74]:


# Assuming pre-trained Word2Vec model is available
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Example text data
sentences = [["hello", "world"], ["word2vec", "model", "training"]]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Save the model
model.save('word2vec_model')




# In[76]:


model


# In[55]:


##Consider Additional Features
#How: Include features like tweet length, hashtags, or user mentions.
df['tweet_length'] = df['text'].apply(len)


# In[56]:


df


# In[57]:


###Model Selection and Training
##Choose Classification Models
#Models: Logistic Regression, Random Forest, Neural Networks, etc.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_reg = LogisticRegression()
rf = RandomForestClassifier()



# In[59]:


log_reg 


# In[60]:


rf


# In[61]:


##Train and Evaluate Models
#How: Fit models on training data and evaluate using cross-validation.
from sklearn.metrics import accuracy_score, classification_report

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))


# In[62]:


##Optimize Hyperparameters
#How: Use grid search or random search to find the best parameters.
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[64]:


X_train


# In[65]:


y_train


# In[66]:


##Part 3:- Model Evaluation and Validation
#Model Evaluation
#Evaluate Metrics
##How: Use metrics like accuracy, precision, recall, F1-score, and confusion matrices.
from sklearn.metrics import confusion_matrix, roc_curve, auc

conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[67]:


##Model Validation
#Validate on Test Data
#How: Ensure the model performs well on unseen data.
y_test_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_test_pred))



# In[68]:


##Check for Overfitting/Underfitting
#How: Compare training and validation performance.
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()


# In[69]:


##Part 4:- Deployment with Web Interface
#Model Deployment
#Serialize the Model
#How: Save the trained model using a format like pickle.
import pickle

with open('disaster_tweet_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)


# In[71]:


pickle.dump


# In[72]:


##Develop Web Application
#How: Use Flask or Django to create a web interface.
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('disaster_tweet_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.json['tweet']
    cleaned_tweet = clean_text(tweet)
    prediction = model.predict([cleaned_tweet])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




