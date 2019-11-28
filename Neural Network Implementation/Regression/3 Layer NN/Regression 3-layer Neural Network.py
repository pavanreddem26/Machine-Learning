#!/usr/bin/env python
# coding: utf-8

# <h1>3 -Layer Neural Network for Linear Regression

# In[1]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


##Relu Output---
def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[3]:


def sigmoid_output(value):
    if(value>=(-709)):
        a=(1+math.exp(-value))
        return(1/a)
    else:
        value=value%100
        a=(1+math.exp(-value))
        return(1/a)


# In[4]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[5]:


def sigmoid_diff(value):
    if(value>=(-709)):
        a=(1 + math.exp(-value))
        b=math.exp(-value)
        return(b/(a*a))
    else:
        value=value%10
        a=(1 + math.exp(-value))
        b=math.exp(-value)
        return(b/(a*a))


# In[6]:


##Change the Path Directory when ever you want--
path=r"C:\Users\Pavan\Desktop\NN testing\regression"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[7]:


# Steps to follow about the data--
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
col_names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
training_data=pd.read_csv("housing_train.txt", sep="\s+", header=None,names=col_names)
training_data  # This gives the total number of the Rows in the Data Set 


# In[8]:


#Separating the label from training data
y=training_data["MEDV"]
y


# In[9]:


#Deleting the label from the training Data
del training_data["MEDV"]
training_data


# In[10]:


#Assuming that the data is completely clean --
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[11]:


total_columns=len(training_data.columns)
total_columns


# In[12]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[13]:


def select_activation(layers):
    
    print("Enter the Activation function for the Hidden Layer")
    print("Enter 'relu' for Rectified Linear Unit")
    print("Enter 'sigmoid' for Sigmoid Activation Unit")

    if(layers==3):
        print("In the 3-layer Neural Network will have only one Hidden Layer and one Activation Function")
        activation_function=input("Enter the activation Function in the hidden Layer")
        output_active=activ_function(activation_function)
        diff_active=diff_activation(activation_function)
        return[output_active,diff_active]
    else:
                
        print("You entered the wrong option")
        print('**Please Start Again**')
        main_function()


# In[14]:


def activ_function(activation_function):
    
    if(activation_function=='relu'):
        
        activ_function=relu_output
        return(activ_function)
    
    elif(activation_function=='sigmoid'):
        
        activ_function=sigmoid_output
        return(activ_function)
    else:
        
        print("You entered the wrong option")
        print('**Please Start Again**')
        main_function()


# In[15]:


def diff_activation(activation_function):
    
    if(activation_function=='relu'):
        
        diff_act=relu_diff
        return(diff_act)
    
    elif(activation_function=='sigmoid'):
        
        diff_act=sigmoid_diff
        return(diff_act)
    else:
        
        print("\nYou entered wrong activation function\n")
        print("**Please Start Again**")
        main_function()


# In[16]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)
    


# In[17]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[18]:


def net_k(hj,W,k,hidden_layer):
    
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[19]:


def update_kj_reg(W,err,hj,LR):
    
    new_updated=W+(LR*(err)*hj)
    return(new_updated)


# In[20]:


def update_ji_reg(V,err,data_point,LR,netj,W,m,diff_active):
    
    ji_updated=(V)+(LR*err*W*diff_active(netj)*data_point[m])
    return(ji_updated)


# In[33]:


#Regression using Batch Gradient descent
def regression_layer_3_batch(layers):
    
    output_layer=1        
    
    input_layer=input_layer_nodes(training_data,"NO")  
    
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
    
    print("\n")
    hidden_activation=select_activation(layers)
    
    output_active=hidden_activation[0]
    
    diff_active=hidden_activation[1]
    
    for l in range(5):

            
        #There are different ways to initialize the weights
        #1) Some random initialization
        #2) Most widely used Xaviers Initialization

        V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
        
        for j in range(hidden_layer):
            
            for m in range(input_layer):
                
                V[j][m]=np.random.uniform(-0.00001,0.00001)
                    
        W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
        
        for k in range(output_layer):
            
            for j in range(hidden_layer):
                
                W[k][j]=np.random.uniform(-0.00001,0.00001)
                    

        LR=(1/(10**(l+1)))
    
        print("Round of LR",(l+1))
        
        print(LR)


        for number in range(100):
            
            print("The round of Iteration:")
            
            t_error=0
                
            error=0 
            
            err=0
            
            for i in range(total_length):
                    
                netj=[0 for m in range(hidden_layer)] 
                
                for j in range(hidden_layer):
                    
                    netj[j]=net_j(training_data.iloc[i],V,j,input_layer)
                
                hj=[0 for n in range(hidden_layer)]
                
                for j in range(hidden_layer):
                    
                    hj[j]=values_hj(output_active, netj[j])
            
                netk=[0 for i in range(output_layer)]
                
                for k in range(output_layer):
                    
                    netk[k]=net_k(hj,W,k,hidden_layer)
                            
                
                for k in range(output_layer):
                    
                    predicted = netk[k]
                    
                    err =(y[i]-predicted) 
                    
                    error=error+(err)
                    
                    t_error=t_error+(err*err)
    
                
            print(number)
            
            print(t_error)
            
            for i in range(total_length):
                
                for j in range(hidden_layer):
                    
                    for m in range(input_layer):
                        
                        for k in range(output_layer):
                            
                            V[j][m]=update_ji_reg(V[j][m],error,training_data.iloc[i],LR,netj[j],W[k][j],m,diff_active)
            
            
            for k in range(output_layer):
                    
                for j in range(hidden_layer):
                        
                    W[k][j]=update_kj_reg(W[k][j],error,hj[j],LR)
            

        print("Total Error(SE):",(t_error))
        print("\n")
        print("The Weights Wkj",W)
        print("\n")
        print("The Weights Vji",V)
        print("\n")
  


# In[39]:


#Regression using Stochastic Gradient descent
def regression_layer_3_stochastic(layers):
    
    output_layer=1                                                          
    
    input_layer=input_layer_nodes(training_data,"NO")  
    
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
    
    print("\n")
    
    hidden_activation=select_activation(layers)
    
    output_active=hidden_activation[0]
    
    diff_active=hidden_activation[1]
    
    inner_iterations=int(input("Enter number of times you want to perform iterations on each data point"))
    
    for l in range(5):
 
        error=0

        #There are different ways to initialize the weights
        #1) Some random initialization
        #2) Most widely used Xaviers Initialization
        V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
        
        for j in range(hidden_layer):
            
            for m in range(input_layer):
                
                V[j][m]=np.random.uniform(-0.00001,0.00001)
        
        W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
        
        for k in range(output_layer):
            
            for j in range(hidden_layer):
                
                W[k][j]=np.random.uniform(-0.00001,0.00001)
                    

        LR=(1/(10**(l+1)))
    
        print("Round of LR:")
        
        print(l+1)
        
        print(LR)

  
        for i in range(len(training_data)):#Length of the data set
        
            print(i)
                    
            err=0  
            
            for p in range(inner_iterations):#In Stochastic Gradient descent we will cnsider each data point for a fixed number of
                                                                     #iterations
                    
                netj=[0 for m in range(hidden_layer)] 
                
                for j in range(hidden_layer):
                    
                    netj[j]=net_j(training_data.iloc[i],V,j,input_layer)
                        
            
                hj=[0 for n in range(hidden_layer)]
                
                for j in range(hidden_layer):
                    
                    hj[j]=values_hj(output_active, netj[j])
            
                netk=[0 for i in range(output_layer)]
                
                for k in range(output_layer):
                    
                    netk[k]=net_k(hj,W,k,hidden_layer)
                            
                
                for k in range(output_layer):
                    
                    predicted = netk[k]
                    
                    err =(y[i]-predicted) 
                
                
                for j in range(hidden_layer):
                    
                    for m in range(input_layer):
                        
                        for k in range(output_layer):
                             
                            V[j][m]=update_ji_reg(V[j][m],err,training_data.iloc[i],LR,netj[j],W[k][j],m,diff_active)
   
                    
            
                for k in range(output_layer):
                    
                    for j in range(hidden_layer):
                        
                        W[k][j]=update_kj_reg(W[k][j],err,hj[j],LR)
                                
                                 

            error=error+(err*err)
        
        print("Total Error:",error)
        
        print("The Weights Wkj",W)
        
        print("The Weights Vji",V)
        
        print("\n")
  


# In[40]:


#Mini Batch Gradient descent
#The main disadvanatge of the Batch Gradient descent is that we may get struck in the local minima(saddle point)
#The stochastic gradient descebt may help us in not getting struck in sadle point but the computational time is very high
#To avoid both these strucking in local minima and Time complexity we will use this technique called MINI BATCH GRADIENT DESCENT

def regression_layer_3_minibatch(layers):
    
    global training_data
    
    output_layer=1                                                          
    
    input_layer=input_layer_nodes(training_data,"NO")  
    
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
    
    print("\n")
    
    hidden_activation=select_activation(layers)
    
    output_active=hidden_activation[0]
    
    diff_active=hidden_activation[1]
    
    batch=int(input('Enter the number of batches you want to divide the data'))
    
    train_batch=[0 for i in range(batch)]
    
    a=0
    
    for i in range(batch):
        
        b=int((((len(training_data))/batch)*(i+1)))
        
        train_batch[i]=training_data[a:b]
        
        a=b
    
    for l in range(5): 
        
        total_error=0

        V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
        
        for j in range(hidden_layer):
            
            for m in range(input_layer):
                
                V[j][m]=np.random.uniform(-0.0001,0.0001)
                    
        W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
        
        for k in range(output_layer):
            
            for j in range(hidden_layer):
                
                W[k][j]=np.random.uniform(-0.0001,0.0001)
            
        LR=(1/(10**(l+1)))

        print("Round of LR",(l+1))

        print(LR)
    
        for j in range(batch):
    
            print(j)
        
            training_data=train_batch[j]
            
            for number in range(50):
                
                t_error=0
                err_1=0
                
                for i in range(len(training_data)):
                    
                        
                    netj=[0 for m in range(hidden_layer)] 
                    
                    for j in range(hidden_layer):
                        
                        netj[j]=net_j(training_data.iloc[i],V,j,input_layer)
                        
            
                    hj=[0 for n in range(hidden_layer)]
                    
                    for j in range(hidden_layer):
                        
                        hj[j]=values_hj(output_active, netj[j])
            
                    netk=[0 for i in range(output_layer)]
                    
                    for k in range(output_layer):
                        
                        netk[k]=net_k(hj,W,k,hidden_layer)
                            
                
                    for k in range(output_layer):
                        
                        predicted = netk[k]
                        
                        err =(y[i]-predicted) 
                        
                        err_1=err_1+(err)
                        
                        t_error=t_error+(err*err)
                        
                for i in range(len(training_data)):
                    
                    for j in range(hidden_layer):
                    
                        for m in range(input_layer):
                        
                            for k in range(output_layer):
                             
                                V[j][m]=update_ji_reg(V[j][m],err_1,training_data.iloc[i],LR,netj[j],W[k][j],m,diff_active)
                
                
                for k in range(output_layer):
                        
                    for j in range(hidden_layer):
                            
                        W[k][j]=update_kj_reg(W[k][j],err_1,hj[j],LR)
                        
            
            
            total_error=total_error+t_error
                                    
            
        print("Total Error:",total_error)
        
        print("The Weights Wkj",W)
        
        print("The Weights Vji",V)
        
        print("\n")
  


# In[41]:


##This is the main Regression Function
def regression_main(Number_of_Layers):
    print("\n The loss for the regression is Root mean Squared Loss\n")
    layers=Number_of_Layers
    if(layers==3):
        print('batch to perform BATCH GRADIENT DESCENT')
        print("stochastic to perform STOCHASTIC GRADIENT DESCENT")
        print("minibatch to perform MINI BATCH GRADIENT DESCENT")
        gd_type=input('Enter the type of Gradient Descent Optimization you want to perform')
        if(gd_type=='batch'):
            regression_layer_3_batch(layers)
        elif(gd_type=='stochastic'):
            regression_layer_3_stochastic(layers)
        elif(gd_type=='minibatch'):
            regression_layer_3_minibatch(layers)
        else:
            main_function()
    else:
        print("\nThe number of layers you wnat to insert is completely incorrect")
        print("\n**Please Try again**")
        main_function()
        
        


# In[42]:


def main_function():
    print("Purpose is Regression \n")
    print("As of now the Number of Layers in neural Network is 3\n")
#The entered input is always considered as string and to change that we need to convert that into 
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Regression" ):
            regression_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")
        print("**Please Start Again**")
        main_function()


# In[46]:


main_function()


# In[44]:


#This is by using the Scikit_learn Library


# In[ ]:


#Using scikit learn library
lin_reg = LinearRegression()
lin_reg.fit(training_data, y)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_


# In[ ]:


train_predictions=lin_reg.predict(training_data)
print("\n The MSE Error for testing is:",mean_squared_error(y,train_predictions))


# In[ ]:





# In[ ]:




