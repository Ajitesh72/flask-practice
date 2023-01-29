import numpy as np
import pandas as pd
import pickle


# %%

# In[3]:


df = pd.read_csv('placement-ml.csv')


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df=df.iloc[:,1:]


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(df['cgpa'],df['iq'],c=df['placement'])


# In[9]:



X = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[10]:


X
y


# In[11]:



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)


# In[12]:



X_test


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler=StandardScaler()


# In[15]:



X_train = scaler.fit_transform(X_train)


# In[16]:


X_train


# In[17]:



X_test = scaler.transform(X_test)


# In[18]:


X_test


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:



clf = LogisticRegression()


# In[21]:


# model training
clf.fit(X_train,y_train)
     


# In[22]:



y_pred = clf.predict(X_test)


# In[23]:
pickle.dump(clf,open('iri.pkl','wb'))



# In[24]:



