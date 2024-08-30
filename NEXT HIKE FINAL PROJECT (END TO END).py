#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 

# In[6]:


import pandas as pd

# Load the datasets
train_data = pd.read_csv('train_data.csv')
test_data_hidden = pd.read_csv('test_data_hidden.csv')
test_data = pd.read_csv('test_data.csv')




# In[7]:


# Display the first few rows of each dataset
print("Train Data:")
print(train_data.head())


# In[8]:


print("\nTest Data Hidden:")
print(test_data_hidden.head())


# In[9]:


print("\nTest Data:")
print(test_data.head())


# In[10]:


#2. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of sentiment categories in the training data
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='sentiment')
plt.title('Distribution of Sentiment Categories')
plt.show()



# In[11]:


#3. Addressing Class Imbalance
from sklearn.utils import resample

# Separate majority and minority classes
majority_class = train_data[train_data.sentiment == 'Neutral']
minority_classes = train_data[train_data.sentiment != 'Neutral']



# In[12]:


# Upsample minority classes
minority_classes_upsampled = resample(minority_classes,
                                      replace=True,
                                      n_samples=len(majority_class),
                                      random_state=123)


# In[13]:


# Combine majority class with upsampled minority classes
train_data_balanced = pd.concat([majority_class, minority_classes_upsampled])


# In[14]:


# Plot the balanced distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data_balanced, x='sentiment')
plt.title('Balanced Distribution of Sentiment Categories')
plt.show()



# In[15]:


#4. Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer





# In[16]:


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(train_data_balanced['reviews.text'])
y = train_data_balanced['sentiment']


# In[17]:


X


# In[18]:


y


# In[19]:


#5. Implementing and Evaluating Classifiers
#5.1. Multinomial Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# In[20]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[21]:


# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


# In[22]:


# Make predictions
y_pred = nb_model.predict(X_test)


# In[23]:


# Evaluation
print("Naive Bayes Model Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[24]:


#**5.2. Multi-class Support Vector Machine (SVM)**

#```python
from sklearn.svm import SVC


# In[25]:


# Train the SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)


# In[26]:


# Make predictions
y_pred_svm = svm_model.predict(X_test)


# In[27]:


# Evaluation
print("SVM Model Evaluation:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# In[28]:


#5.3. Neural Networks (LSTM) 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[29]:


# Load datasets
train_data = pd.read_csv('train_data.csv')
train_data


# In[30]:


# Encode sentiment labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(train_data['sentiment'])


# In[31]:


# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['reviews.text'])
X_sequences = tokenizer.texts_to_sequences(train_data['reviews.text'])
X_padded = pad_sequences(X_sequences, maxlen=100)


# In[32]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.3, random_state=123)


# In[33]:


# Define LSTM model
model = Sequential([
    Embedding(5000, 128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dense(len(label_encoder.classes_), activation='softmax')
])


# In[34]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[35]:


# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))


# In[36]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM Model Loss: {loss}')
print(f'LSTM Model Accuracy: {accuracy}')


# In[37]:


# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)


# In[38]:


# Display some predictions
print("\nSample Predictions:")
print(pd.DataFrame({'Actual': label_encoder.inverse_transform(y_test), 'Predicted': y_pred_labels}).head())


# In[39]:


#6. Model Optimization
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 



# In[40]:


# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}


# In[41]:


# Initialize SVM
svm = SVC(probability=True)

# Initialize GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=2)


# In[ ]:


# Fit GridSearchCV
grid_search.fit(X_train, y_train)


# In[ ]:


# Print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')


# In[ ]:





# In[ ]:


# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Preprocess the data
# Using CountVectorizer for LDA instead of TF-IDF
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_count = count_vectorizer.fit_transform(train_data_balanced['reviews.text'])

# Fit LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda_model.fit_transform(X_count)

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

# Display top words per topic
display_topics(lda_model, count_vectorizer.get_feature_names_out(), 10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




