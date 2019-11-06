
# coding: utf-8

# <h3> Implementing Perceptron Algorithm

# <h5> Assumption in Perceptron Algorithms:<br>
# <br><br>
# <b>The Data should be Linearly Separable<br><br><br>
# <b> The hyperplane or the line should pass through the origin<br>

# In[1]:


import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


# In[2]:


#checking the path directory
print("Previous directory",os.getcwd())
#Now changing the path directory
path=r"C:\Users\Pavan\Desktop\Machine Learning Github\Home Works\HW2"
os.chdir(path)
print("Current directory",os.getcwd())


# In[3]:


col=['a','b','c','d','y']
perceptron=pd.read_table("perceptrons.txt",header=None,names=col)
perceptron


# In[4]:


#First thing you have to do in Perceptrons is to change all the labels into a 1 class
for i in range(len(perceptron)):
    if(perceptron['y'].iloc[i]==(-1)):
        perceptron.iloc[i]=(-1)*perceptron.iloc[i]


# In[5]:


#Now you can see that all the values are converted into the Positive Class
perceptron


# In[6]:


len(perceptron)


# In[7]:


y=perceptron['y']
y


# In[8]:


del perceptron['y']
perceptron


# In[9]:


#Now Normalizing the perceptron Data i.e the value sg


# In[10]:


total_number_weights=(len(perceptron.columns))
total_number_weights


# In[11]:


#First initialize then to some random values
w=[0 for i in range(total_number_weights)]
for i in range(total_number_weights):
    w[i]=np.random.uniform(-20,20)

w


# In[12]:


w_new=np.reshape(w,(4,1))
w_new


# In[13]:


##since in perceptrons we dont have any closed from we can solve them only by using Iteration Methods

n=int(input("Enter the Number of Iterations you want to perform\n"))

for i in range(n):
    
    count_misclassified=0
    
    for j in range(len(perceptron)):
        
        
        
        out=((perceptron.iloc[j]).dot(w_new))#since this value is 1 and -1
        
        if(out<=0):
            
            count_misclassified=count_misclassified+1
            
            for k in range(total_number_weights):
                
                w_new[k]=w_new[k]+(0.0005*(perceptron.iloc[j][k]))
           
     
    print("Iteration:",i)
    print("Missclassified:",count_misclassified)


# <h5>Practically speaking we always need to find the right combination of w's and learning rate
# <h5>If the learning rate is very less then the number of iterations should be increased to find the Convergence
# <h5>If the learning rate is high then it is diificult to find the convergence because this moves like crazy
# <h5>So It comes down to finding the appropriate learning rate
# 

# <h3> You can see that as the number of iterations are increasing the number of points that are misclassified gradually decreases
# <h3>If the data is Linealy Separable then you can find the hyperplane that has zero misclassification(Theoretically Speaking)
# 

# In[15]:


w_new

