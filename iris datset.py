#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[14]:


import numpy as np


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


import seaborn as sns


# In[217]:


from sklearn import metrics


# In[17]:


from sklearn.datasets import load_iris


# In[31]:


import sklearn


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[164]:


iris = pd.read_csv('C:/Users/heyit/OneDrive/Desktop/Iris.csv')


# In[ ]:





# In[165]:


# print the names of the four features
iris.feature_names


# In[220]:


iris.head()


# In[169]:


iris.info()


# In[172]:


iris.drop('Id',axis=1,inplace=True)


# In[ ]:





# In[170]:


iris.target_names


# In[173]:


iris.shape


# In[176]:


iris.target.shape


# In[177]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[178]:


fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[180]:


iris.hist(edgecolor='red', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()


# In[130]:


# store feature matrix in "X"
X = iris.data

# store response vector in "Y"
Y = iris.target


# In[189]:


iris.head()


# In[190]:


from sklearn.model_selection import train_test_split


# In[191]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[194]:


Y_test.head(2)


# In[183]:



mean_x = np.mean(X)
mean_y = np.mean(Y)


# In[184]:


mean_x


# In[185]:


mean_y


# In[139]:


max=np.amax(iris ,axis=0)


# In[140]:


max


# In[141]:


min=np.amin(iris ,axis=0)


# In[142]:


min


# In[196]:


rows,col=iris.shape
print("Rows:%s,column:%s"%(rows, col))


# In[197]:


from sklearn.linear_model import LinearRegression


# In[198]:


lr = LinearRegression()


# In[199]:


lr.fit(X_train,Y_train)


# In[200]:


pred = lr.predict(X_test)


# In[201]:


plt.scatter(Y_test,pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Predicted value-true value=Linear Regression")

plt.show()


# In[202]:


#Total number of values


# In[203]:


n=len(X)


# In[204]:


num=0
deno=0
for i in range(n):
    num+=(X[i]-mean_x)-(Y[i]-mean_y)
    deno+=(X[i]-mean_x)**2
    m=num/deno
    c=mean_y-(m*mean_x)
    print(m,c)


# In[205]:


max_x=np.max(X)+100
min_x=np.min(X)+100


# In[206]:


#calculating line values of x and y


# In[207]:


x=np.linspace(min_x,max_x,1000)


# In[208]:


y = c + m*x


# In[209]:


print(m,c)


# In[214]:


from sklearn.linear_model import LogisticRegression


# In[219]:


model = LogisticRegression()
model.fit(X_train,Y_train)
prediction=model.predict(X_test)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,Y_test))


# In[222]:


print(iris)


# In[ ]:




