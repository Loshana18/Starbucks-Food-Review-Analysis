#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

import string
import re
from collections import Counter

import nltk
import transformers


# # Importation of data and EDA 

# In[2]:


df = pd.read_csv('D:/DATA CASE STUDY/reviews_data.csv')
df.head()


# In[3]:


df.info()


# In[4]:


#Dimension of the data.
df.shape


# In[5]:


df.columns.tolist()


# In[6]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.set()
plt.style.use('ggplot')


# In[10]:


#Checking for null values
df.isnull().sum()


# In[9]:


# Let's exclude null values.
df = df.dropna()


# In[11]:


# Check the unique values and frequency for 'Rating'

df['Rating'].value_counts()


# In[12]:


# dropping links
df.drop('Image_Links', axis=1, inplace=True)
df.head()


# In[13]:


# remove entries with no review
df = df[df['Review'] != 'No Review Text']
df = df.reset_index(drop=True)
df.head()


# In[15]:


# Importing necessary module
from dateutil import parser

# expanding date
df['Date'] = df['Date'].str.replace('Reviewed', '').str.replace('[.,]', '')

# parsing dates into standard form
df['Date'] = df['Date'].apply(parser.parse)

# segregating dates
df['month'] = df['Date'].dt.strftime('%b')
df['year'] = df['Date'].dt.year
df['date'] = df['Date'].dt.day

# dropping old date column
df.drop('Date', axis=1, inplace=True)

df.head()


# In[16]:


# renaming columns
df.columns = ['name','location','rating','review','month','year','date']

# rearanging columns
df = df[['name','location','date','month','year','review','rating']]

df.head()


# In[17]:


# function to remove unsupported characters from reviews
def clean_review(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())

# removing unupported characters
df['review'] = df['review'].apply(clean_review)

# making all statements sentence case
df['review'] = df['review'].apply(lambda x: x.capitalize())

# viewing data
df.head()


# # Visualizations

# In[19]:


freq = df['rating'].value_counts()
freq.plot(kind='bar', title='Frequency of ratings', xlabel = 'rating', ylabel = 'Frequency')
plt.grid()
plt.show()


# In[20]:


# Locations of the interviewees

print('There are ',len(df.location.unique()),'locations in our dataset')

locations_count =  df.location.value_counts().sort_values(ascending = False) [:10]

# ploting the top 10 locations
plt.barh(locations_count.index ,locations_count.values)
plt.title('Most frequent locations in our dataset')
plt.tight_layout()
plt.show()


# In[24]:


# Let's clean the data

from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


# In[25]:


nltk.download('all')
get_ipython().system('unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/')


# In[27]:


text = ' '.join(df['review'])


# In[28]:


def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]+', ' ',text)
    text = re.sub('[%s]'%re.escape(string.punctuation) ,' ',text)
    text =  ' '.join(lemmatizer.lemmatize(stemmer.stem(word)) for word in text.split(' ') if word not in stop_words)
    return text


# In[62]:


df['clean_review'] = df['review'].progress_apply(clean_text)
df.head()


# In[31]:


get_ipython().system('pip install wordcloud')


# In[66]:


all_words_list  = df['clean_review'].str.split().sum()
count = Counter( all_words_list ) 
word_dict = dict( sorted(count.items() , key = lambda k:k[1] , reverse = True) ) 

fig  = px.bar(x = list(word_dict.keys())[:20] ,
              y = list(word_dict.values())[:20] ,
              color = list(word_dict.values())[:20])

fig.update_layout(title_text= 'Top 20 most common words in our dataset' , xaxis = dict(title = 'Words'), yaxis = dict(title = 'Count') , showlegend=False)
fig.show()


# In[64]:





# In[ ]:





# # Sentiment Analysis

# In[34]:


# Importing the Vader sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# In[35]:


# test
sia.polarity_scores('I am doing good')


# In[37]:


# Applying the vader SA on all the dataset

res = {}
for i,row in tqdm(df.iterrows() , total = len(df)):
    text = row['clean_review']
    myname = row['name']
    res[myname] = sia.polarity_scores(text)


# In[38]:


#Stocking the results in a dataframe
vader_df = pd.DataFrame(res).T
vader_df.head()


# In[67]:


df = df.merge(vader_df.reset_index(names = 'name')  , how = 'left')
df.head()


# In[40]:


import seaborn as sns


# In[68]:


## Analyzing of the scores

# Representing the compound score by ratings

sns.barplot(data = df ,x = 'rating' , y = 'compound')
plt.title('Compound scores by ratings')
plt.show()


# In[56]:


# Positive , Neutral and Negative score by Rating

fig , axs = plt.subplots(1,3,figsize = (16,6))
sns.barplot(data = df ,x = 'rating' , y = 'pos' , ax = axs[0])
axs[0].set_title('Positives scores by Rating')
sns.barplot(data = df ,x = 'rating' , y = 'neu' , ax = axs[1])
axs[1].set_title('Neutral scores by Rating')
sns.barplot(data = df ,x = 'rating' , y = 'neg', ax = axs[2])
axs[2].set_title('Negatives scores by Rating')
plt.show()


# # EDA

# # Rating
# 

# In[71]:


df["rating"]=df["rating"].replace([1,2],2)
df["rating"]=df["rating"].replace(3,0)
df["rating"]=df["rating"].replace([4,5],1)
df = df.dropna(axis = 0, how ='any')
df["rating"]=df["rating"].astype(int)


# In[72]:


df["rating"].value_counts()


# In[73]:


plt.figure(figsize=(6, 6))
sns.set_style("whitegrid")
plt.pie(df["rating"].value_counts(),labels=["Negative","Positive","Neural"], autopct='%1.1f%%',colors=['#66b3ff', '#99ff99', '#ffcc99'] , startangle=90);


# # Location

# In[ ]:





# In[75]:


plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
df["location"].value_counts().sort_values(ascending= False).head(10).plot.bar(color='skyblue')
plt.title("Count Of Reviews per State");


# # Months

# In[79]:


df.head()


# In[82]:


df = df.groupby(['month'])['rating'].sum().reset_index()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.figure(figsize=(10, 6))
plt.plot(df['month'], df['rating'], marker='o', color='skyblue', linestyle='-')
plt.xlabel('month')
plt.ylabel('Total Rating')
plt.title('Total Rating Count Per Month')

plt.xticks(df['month'], labels=month_order,rotation=45)
plt.grid(True)
plt.show()


# In[ ]:




