#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#Data set from csv to Pandas DataFrame


# In[3]:


titanic_data = pd.read_csv('titanic_train.csv')


# In[4]:


titanic_data.head()


# In[5]:


titanic_data.tail()


# In[6]:


# numbers of row and colums
titanic_data.shape


# In[7]:


titanic_data.info()


# In[8]:


titanic_data.isnull().sum()


# In[9]:


#Handle the missing value

titanic_data = titanic_data.drop(columns='Cabin', axis=1) #drop the cabin colum from the dataframe


# In[10]:


# replacing the missing values in Age colum
titanic_data['Age'].fillna(titanic_data['Age'].mean,inplace=True)


# In[11]:


# finding the mode value of 'Embarked' colum
print(titanic_data['Embarked'].mode())


# In[12]:


print(titanic_data['Embarked'].mode()[0])


# In[13]:


# replacing the missing values in 'Embark' colum with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[14]:


titanic_data.isnull().sum()    #cheak missing value


# # Data Analisys

# In[15]:


titanic_data.describe()


# In[16]:


titanic_data['Sex'].value_counts()


# # Data Visualization

# In[17]:


sns.set()


# In[18]:


# makig a count plot for "Survied" colum
sns.countplot('Survived', data=titanic_data)


# In[19]:


# makig a count plot for "Sex" colum
sns.countplot('Sex', data=titanic_data)


# In[20]:


# number of suvivers Genders wise
sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[21]:


# makig a count plot for "Pclass" colum
sns.countplot('Pclass', data=titanic_data)


# In[22]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# In[23]:


#Encoding the Categorial Colums
titanic_data['Sex'].value_counts()


# In[24]:


titanic_data['Embarked'].value_counts()


# In[25]:


# converting Categorial Colums
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[26]:


titanic_data.head()


# In[27]:


X = titanic_data.drop(columns = ["PassengerId","Name","Ticket","Survived","SibSp","Parch"],axis=1)
Y = titanic_data['Survived']


# In[28]:


titanic_data.head(5)


# In[35]:


feature_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
X = titanic_data[feature_cols]
y = titanic_data[["Survived"]]


# In[36]:


X.info()


# In[52]:


y.info()


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)


# In[60]:


print(X.shape, X_train.shape, X_test.shape)


# In[74]:


print(y.shape, y_train.shape, y_test.shape)


# In[61]:


# Logistic Regression


# In[77]:


model = LogisticRegression()


# In[ ]:


#   Model Evolution


# In[ ]:


# Accurecy on training data
X_train_prediction = model.predict(X_train)


# In[ ]:


print(X_train_prediction)


# In[ ]:


training_data_accuracy = accuracy_score(y_train,X_train_prediction)
print('Accuracy score of training data: ', training_data_accuracy)


# In[ ]:


#trainning the Logistic Regression model with training data
model.fit(X_train, y_train)


# In[ ]:


# Accurecy on test data
X_test_prediction = model.predict(X_test)


# In[ ]:


print(X_test_prediction)


# In[ ]:


test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accurecy score of test data :' , test_data_accuracy)

