#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install seaborn


# In[2]:


pip install numpy


# In[3]:


pip install pandas


# In[4]:


pip install scikit-learn


# In[5]:


pip install xgboost


# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# In[7]:


calories = pd.read_csv('calories.csv')


# In[8]:


calories.head()


# In[9]:


exercise = pd.read_csv('exercise.csv')


# In[10]:


exercise.head()


# In[11]:


calories = calories.merge(exercise,on='User_ID')


# In[12]:


calories.head()


# In[13]:


calories.shape


# In[14]:


calories.isnull().sum()


# In[15]:


sns.set()


# In[16]:


sns.countplot(x='Gender', data=calories)


# In[17]:


sns.distplot(calories['Age'])


# In[18]:


sns.distplot(calories['Height'])


# In[19]:


sns.distplot(calories['Weight'])


# In[20]:


correlation = calories.corr()


# In[21]:


plt.figure(figsize=(12,12))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[22]:


calories.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[23]:


calories.head()


# In[24]:


X = calories.drop(columns = ['User_ID','Calories'],axis=1)
Y = calories.Calories


# In[25]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


from sklearn.model_selection import RandomizedSearchCV


# In[28]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[29]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[30]:


modell1 = XGBRegressor()
model1 = RandomizedSearchCV(estimator = modell1, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
model1.fit(X_train, Y_train)


# In[31]:


predictions1 = model1.predict(X_test)


# In[32]:


plt.scatter(Y_test,predictions1)


# In[33]:


mae = metrics.mean_absolute_error(Y_test,predictions1)
print("Mean Absolute Error = ", mae)


# In[34]:


mse = metrics.mean_squared_error(Y_test,predictions1)
print("Mean Squared Error = ", mse)


# In[35]:


r2 = metrics.r2_score(Y_test,predictions1)
print("R2 score = ", r2)


# In[36]:


model2 = LinearRegression()
model2.fit(X_train, Y_train)


# In[37]:


predictions2 =model2.predict(X_test)


# In[38]:


plt.scatter(Y_test,predictions2)


# In[39]:


mae = metrics.mean_absolute_error(Y_test,predictions2)
print("Mean Absolute Error = ", mae)


# In[40]:


mse = metrics.mean_squared_error(Y_test,predictions2)
print("Mean Squared Error = ", mse)


# In[41]:


r2 = metrics.r2_score(Y_test,predictions2)
print("R2 score = ", r2)


# In[42]:


gender = input("Enter your gender (M/F): ")
age = int(input("Enter your age: "))
height = float(input("Enter your height (in centimeters): "))
weight = float(input("Enter your weight (in kg): "))
duration = int(input("Enter the duration of exercise (in minutes): "))
heart_rate = int(input("Enter your heart rate (in beats per minute): "))
body_temp = float(input("Enter your body temperature (in Celsius): "))

gender = 1 if gender.upper() == 'F' else 0

input_data = (gender,age,height,weight,duration,heart_rate,body_temp)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction1 = model1.predict(input_data_reshaped)

print('The calories burned are somewhat around ', round(prediction1[0],0))


# In[44]:


import pickle

file = open('model.pkl', 'wb')

pickle.dump(model1, file)

