
# coding: utf-8

# In[254]:


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[255]:


#checking the path directory
print("Previous directory",os.getcwd())
#Now changing the path directory
path=r"C:\Users\Pavan\Desktop\Machine Learning Github\Home Works\HW1"
os.chdir(path)
print("Current directory",os.getcwd())


# Hosuing data set ----Question 1
# 
# Run linear regression using Normal Equations and Mean Square Error without using the libraries

# In[256]:


#Loading the DataSet
col_names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
train_housing=pd.read_table("housing_train.txt",sep="\s+",names=col_names)
train_housing


# In[257]:


y_train=train_housing["MEDV"]
y_train


# In[258]:


del train_housing["MEDV"]
train_housing


# In[259]:


#Using the Normal Equations I am calculating the coefficients of the paremets(w1,w2,w3-----w13)
#Now we have the Data matrix(X) and the labels(Y) now we need to find the coefficients
w_normal_equations=np.linalg.inv(train_housing.T.dot(train_housing)).dot(train_housing.T).dot(y_train)
w_normal_equations


# In[260]:


w=np.reshape(w_normal_equations,(1,13))
w


# In[261]:


##Function for calculating mean squared error using the Normal Equations for the training set
def mean_square_error(y,w,x,n_iterations):
    squared_error=0
    mean_squared_error=0
    for i in range(n_iterations):#For all the data points considerong the (Y-h(x))i.e(y-WX)
        squared_error=squared_error+((y_train[i]-(w.dot(x.iloc[i])))*(y[i]-(w.dot(x.iloc[i]))))
    mean_squared_error=squared_error/n_iterations
    
    print("Mean Squared Error for the training is:",mean_squared_error)

        
mean_square_error(y_train,w,train_housing,433)


# In[262]:


#Now we are conisdering the testing data set
test_data=pd.read_table("housing_test.txt",sep="\s+",header=None,names=col_names)
test_data


# In[263]:


y_test=test_data["MEDV"]
y_test


# In[264]:


del test_data["MEDV"]


# In[265]:


x_test=test_data
x_test


# In[266]:


##The mean squared error function is already defined before and I just called that function using the function name
mean_square_error(y_test,w,x_test,74)


# <h5>train error: 24.47
# <h5>test error: 24.29
# <h5>using the Normal Equations

# <h3>Part 2 : Perform everything using the ScikitLearn package

# In[270]:


##Mean squared error using Scikit Learn for the training data
lin_reg = LinearRegression()
lin_reg.fit(train_housing, y_train)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_
train_predictions=lin_reg.predict(train_housing)
print("\n The MSE Error for testing is:",mean_squared_error(y_train,train_predictions))


# In[272]:


##Now calculating the mean sqaured error for the testing dataset
test_predictions=lin_reg.predict(test_data)
print("\n The MSE Error for the testing is:",mean_squared_error(y_test,test_predictions))


# <h5>Training error: 22.08
# <h5>Testing error: 22.63
# <h5>using the scikitlearn package

# <h3>NOW PERFORMING EVERYTHING BY NORMALIZING the Data

# In[273]:


train_housing


# In[274]:


#This gives the total Number of time syou need to learn the inner loop
x=len(train_housing)
x


# In[275]:


#This give sthe total number of times you need to run the for loop
y=len(train_housing.columns)
y


# In[276]:


#Function for Normalization
#X-Number of Rows
#Y- Number of columns
def Normalization(train_housing,x,y):
    for i in range(y):
        z=train_housing.columns[i]
        for j in range(x):
            train_housing[z].iloc[j]=(((train_housing[z].iloc[j]-(min(train_housing[z])))/(max(train_housing[z])-min(train_housing[z]))))


# In[277]:


#This is the Normalized data(Training Data)
Normalization(train_housing,x,y)


# In[278]:


#This is Normalizing the test data
x=len(test_data)
y=len(test_data.columns)
Normalization(x_test,x,y)


# In[279]:


x_test


# In[280]:


train_housing.shape


# In[281]:


y_train.shape


# In[282]:


w_equations=np.linalg.inv(train_housing.T.dot(train_housing)).dot(train_housing.T).dot(y_train)
w_equations


# In[283]:


w_new=w_equations.reshape(1,13)
w_new


# In[284]:


#This is forthe training data
mean_square_error(y_train,w_new,train_housing,433)


# In[285]:


#Now calculating the error for the test data
mean_square_error(y_test,w_new,x_test,74)


# <h5> Results:
# <h5> Training Error: 26.33
# <h5> Testing Error:  32.104

# In[286]:


#by using the scikit learn library
lin_reg = LinearRegression()
lin_reg.fit(train_housing, y_train)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_
train_predictions=lin_reg.predict(train_housing)
print("\n The MSE Error for testing is:",mean_squared_error(y_train,train_predictions))


# In[287]:


##Now calculating the mean sqaured error for the testing dataset
test_predictions=lin_reg.predict(x_test)
print("\n The MSE Error for the testing is:",mean_squared_error(y_test,test_predictions))


# <h5> By using library:
# <h5> Training Error: 25.1277
# <h5> Testing Error:  30.2312
