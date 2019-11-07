
# coding: utf-8

# <h2> Question-1A
# 

# <h3>Linear Regression and Ridge Regression Using Normal Equations and Gradient Descent Iteration Method

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


#checking the path directory
print("Previous directory",os.getcwd())
#Now changing the path directory
path=r"C:\Users\Pavan\Desktop\Machine Learning Github\Home Works\HW1"
os.chdir(path)
print("Current directory",os.getcwd())


# In[3]:


col=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
train_housing=pd.read_table("housing_train.txt",sep='\s+',names=col)
train_housing


# In[4]:


y_train=train_housing['MEDV']
y_train


# In[5]:


del train_housing['MEDV']
train_housing


# In[6]:


#Now training using Regression Normal Equations an Ridge Regression Normal Equations

w_normal=np.linalg.inv((train_housing.T).dot(train_housing)).dot(train_housing.T).dot(y_train)
w_normal  #Find the coefficients using the Linear Regression Normal Equations


# In[7]:


#Now training the same data using the Ridge Regression Normal Equation
#The main advantage of Ridge Regression over the Normal Regression is that RESTRICTION on w's
#In this Ridge regression the w's are distributed(as opposed to Linear Regression)
#As teh apha value Increases we are restricting the w values more and more 

n=int(input("Enter the number of alpha values you want to try\n"))
alpha=[0 for i in range(n)]

for i in range(n):
    print('\n')
    print(i)
    alpha[i]=float(input("Enter the value of alpha\n"))

#From the above loop we got the values of alpha 
    
w_ridge=[[0 for i in range(len(train_housing.columns))]for i in range(n)]


for i in range(n):
    
    w_ridge[i]=np.linalg.inv(((train_housing.T).dot(train_housing))+(alpha[i]*np.identity(len(train_housing.columns)))).dot(train_housing.T).dot(y_train)
    


# In[8]:


w_ridge


# In[9]:


#Till now I trained the model but now I need to test and find the error
def mean_squared_error(train,y,w,length):
    squared_error=0
    for i in range(length):
        predicted=(w.dot(train_housing.iloc[i].T))
        error=(y[i]-predicted)
        squared_error=squared_error+(error*error)
    return(squared_error/length)


# In[10]:


#Training Error
#This is the mean sqaured error using the Linear regression Normal Equations
mse_train_linear=mean_squared_error(train_housing,y_train,w_normal,len(train_housing))
mse_train_linear


# In[11]:


#Training Error
#Now find the mean Sqaured error using the Ridge Regression Normal Equations
mse_train_ridge=[0 for i in range(n)]
for i in range(n):
    mse_train_ridge[i]=mean_squared_error(train_housing,y_train,w_ridge[i],len(train_housing))
mse_train_ridge


# In[12]:


#Testing Error
test_housing=pd.read_csv("housing_test.txt",sep="\s+",header=None,names=col)
test_housing


# In[13]:


y_test=test_housing["MEDV"]
y_test


# In[14]:


del test_housing["MEDV"]
test_housing


# In[15]:


#Since I already trained the model now I can test directly
mse_test_linear=mean_squared_error(test_housing,y_test,w_normal,len(test_housing))
mse_test_linear


# In[16]:


mse_test_ridge=[0 for i in range(n)]
for i in range(n):
    mse_test_ridge[i]=mean_squared_error(test_housing,y_test,w_ridge[i],len(test_housing))
mse_test_ridge


# In[17]:


print("Training Errors:")
print("Using Linear Regression:",mse_train_linear)
print("Using Ridge Regression(with different ALPHA values):",mse_train_ridge)

print("Testing Errors:")
print("Using Linear Regression:",mse_test_linear)
print("Using Ridge Regression(with different ALPHA values):",mse_test_ridge)


# <h3> Linear Regression and Logistic Regression using Gradient descent Iteration method

# In[18]:


#Linear Regression uisng Gradient Descent Update rule
#First Intialize the weights to some random values
w_reg=[0 for i in range(len(train_housing.columns))]
for i in range(len(train_housing.columns)):
    w_reg[i]=np.random.uniform(0,0)
#Intialized the values to some random values


# In[19]:


w=np.reshape(w_reg,((len(train_housing.columns)),1))
w


# In[20]:


LR=0.00001


# In[21]:


#Gradient Descent Iteration method for Linear Regression
#This is mainly for finding the weights(w) using iteration method as opposed to Normal Equations

n=int(input("Enter the number of iterations you want to perform\n"))

for i in range(n):
    
    error=0
    
    for j in range(len(train_housing)):
        
        output=((train_housing.iloc[j].T).dot(w))
        err=((y_train[j]*1000)-output)
        error=error+(err*err)
        for k in range(len(train_housing.columns)):
            w[k]=w[k]-(LR*err*train_housing.iloc[j][k])
    print("Round:",i)
    print("Error:",error/len(train_housing))


# In[ ]:


print(w)


# In[ ]:


##Ridge Regression using Gradient Descent Iteration Method
#Alpha is generally a langrange Constraint and the optimization we are finding is the Constraint Optimization

Alpha=float(input("Enter the value of the Alpha in the ridge regression"))

n_1=int(input("Enter the number of iterations you want to perform\n"))

w_rid=[0 for i in range(len(train_housing.columns))]
#Now intialize the values to some random values
for i in range(len(train_housing.columns)):
    w_rid[i]=np.random.uniform(-10,10)
    



# In[ ]:


#Changing the Shape of the w's so that it can support Matrix Multiplication
w_rid_gd=np.reshape(w_rid,((len(train_housing.columns)),1))
w_rid_gd


# In[ ]:


#Gradient Descent Rule for Ridge Regression
for i in range(n_1):
    
    error=0
    
    for j in range(len(train_housing)):
        
        output=((train_housing.iloc[j]).dot(w_rid_gd))
        
        err=((y_train[j]*1000)-output)
        
        print(err)
        
        error=error+(err*err)
        
        for k in range(len(train_housing.columns)):
            
            w_rid_gd[k]= w_rid_gd[k]-(LR*Alpha*w_rid_gd[k])-(LR*(err)*train_housing.iloc[j][k])
    
    print("Round:",i)
    print("Error:",error/(len(train_housing)))


# In[ ]:


w_rid_gd

