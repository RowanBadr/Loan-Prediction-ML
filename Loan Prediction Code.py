#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[60]:


df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
df.head(10)


# In[61]:


df = df.fillna(value={'LoanAmount' :df['LoanAmount'].mean()})
df = df.fillna(value={'CoapplicantIncome' :df['CoapplicantIncome'].mean()})
df = df.fillna(value={'ApplicantIncome' :df['ApplicantIncome'].mean()})


# In[62]:





# In[95]:


df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['Married'].replace(['Yes', 'No'], [0,1], inplace = True)
df['Education'].replace(['Graduate', 'Not Graduate'], [0,1], inplace = True)
df['Self_Employed'].replace(['No', 'Yes'], [0,1], inplace = True)
df['Property_Area'].replace(['Urban', 'Rural', 'Semiurban'], [0,1,2], inplace = True)
df['Loan_Status'].replace(['N', 'Y'], [0,1], inplace = True)


# In[96]:


df = df.dropna(axis = 0, how= 'any')
df.isnull().sum()


# In[97]:


y=df['Loan_Status']
x=df[['Gender', 'Married', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History', 'Property_Area']]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size=0.25)


# In[105]:


model = DecisionTreeClassifier(max_depth = 4)
model.fit(x_train, y_train)


# In[106]:


pred = model.predict(x_test)
print(accuracy_score(y_test, pred))


# In[100]:


modelb = SVC()
modelb.fit(x_train, y_train)


# In[101]:


predb = modelb.predict(x_test)
print(accuracy_score(y_test, predb))


# In[102]:


modela = KNeighborsClassifier()
modela.fit(x_train, y_train)


# In[103]:


preda = modela.predict(x_test)
print(accuracy_score(y_test, preda))

