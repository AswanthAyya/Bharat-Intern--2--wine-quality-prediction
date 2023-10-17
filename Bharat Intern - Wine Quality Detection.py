#!/usr/bin/env python
# coding: utf-8

#  <center> 
#    -: Wine Quality Prediction :- <br>
# Machine Learning model to predict the quality of wine using linear regression <br>
#    - sumitted by Muni Aswanth Prasad A
# </center>

# ### Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Import the Dataset and Pre-Processing the Data

# In[2]:


data = pd.read_csv('wine.csv')


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().any()


# In[8]:


data.isnull().sum()


# In[9]:


data.columns


# In[10]:


plt.figure(figsize=(50,5))
sns.countplot(data)


# In[14]:


g = sns.PairGrid(data)
g.map_upper(sns.scatterplot,color='Purple')
g.map_lower(sns.scatterplot, color='Black')
g.map_diag(plt.hist,color='#0146B6')


# In[15]:


sns.histplot(data['quality'],color="purple")


# In[16]:


corr=data.corr()
corr


# In[17]:


plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True)


# ## Taining the model

# In[19]:


X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
y = data['quality']


# #  Train and Test Split

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Training Model

# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lm = LinearRegression()


# In[24]:


lm.fit(X_train,y_train)


# ## Model Evaluation 

# In[25]:


print(lm.intercept_)


# In[26]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[27]:


predictions = lm.predict(X_test)


# In[28]:


plt.scatter(y_test,predictions,color="purple")


# In[29]:


sns.displot((y_test-predictions),bins=50,color="purple");


# In[30]:


from sklearn import metrics


# In[31]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[34]:


data.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# In[ ]:




