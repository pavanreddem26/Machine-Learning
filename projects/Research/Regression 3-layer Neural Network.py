
# coding: utf-8

# <h1>3 -Layer Neural Network for Linear Regression

# In[101]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[102]:


##Relu Output---
def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[103]:


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[104]:


def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[105]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[106]:


def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[107]:


def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[108]:


##Change the Path Directory when ever you want--
path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[109]:


# Steps to follow about the data--
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 


# In[110]:


#Separating the label from training data
y=training_data[3]
y


# In[111]:


#Deleting the label from the training Data
del training_data[3]
training_data


# In[112]:


#Assuming that the data is completely clean --
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[113]:


total_columns=len(training_data.columns)
total_columns


# In[114]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[115]:


def select_activation(layers):
    
    print("Enter the Activation function for the Hidden Layer")
    print("Enter 'relu' for Rectified Linear Unit")
    print("Enter 'sigmoid' for Sigmoid Activation Unit")
    print("Enter 'linear' for Linear Activation Function")
    
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


# In[116]:


def activ_function(activation_function):
    
    if(activation_function=='relu'):
        
        activ_function=relu_output
        return(activ_function)
    
    elif(activation_function=='sigmoid'):
        
        activ_function=sigmoid_output
        return(activ_function)
    
    elif(activation_function=='linear'):
        
        activ_function=linear_output
        return(activ_function)
    
    else:
        
        print("You entered the wrong option")
        print('**Please Start Again**')
        main_function()


# In[117]:


def diff_activation(activation_function):
    
    if(activation_function=='relu'):
        
        diff_act=relu_diff
        return(diff_act)
    
    elif(activation_function=='sigmoid'):
        
        diff_act=sigmoid_diff
        return(diff_act)
    
    elif(activation_function=='linear'):
        
        diff_act=linear_diff
        return(diff_act)
    
    else:
        
        print("\nYou entered wrong activation function\n")
        print("**Please Start Again**")
        main_function()


# In[118]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)
    


# In[119]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[120]:


def net_k(hj,W,k,hidden_layer):
    
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[121]:


def update_kj_reg(W,err,hj,LR):
    
    new_updated=W+(LR*(err)*hj)
    return(new_updated)


# In[122]:


def update_ji_reg(V,err,data_point,LR,netj,W,m,diff_active):
    
    ji_updated=(V)+(LR*err*W*diff_active(netj)*data_point[m])
    return(ji_updated)


# In[127]:


def regression_layer_3(layers):
    output_layer=1                                                          
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
    print("\n")
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]
    diff_active=hidden_activation[1]
    
    err_q=[[0 for i in range(5)]for y in range(5)]
    W_q=[[0 for i in range(5)]for y in range(5)]
    V_q=[[0 for i in range(5)]for y in range(5)]
    LR=[[0 for i in range(5)]for y in range(5)]
    
    for q in range(5):
        
        
        for l in range(5):
        
            print("\n")
            print("Round:",q)
            print("\n")
            
            #There are different ways to initialize the weights
            #1) Some random initialization
            #2) Most widely used Xaviers Initialization

            V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
            W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
            for k in range(output_layer):
                for j in range(hidden_layer):
                    W[k][j]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
                    
            
            LR[q][l]=(1/(10**(l+2)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[q][l])
            print("\n")

            
            
            for number in range(100):
                
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
                        error=error+((y[i]-predicted)*(y[i]-predicted))
            
                

                    for k in range(output_layer):
                        for j in range(hidden_layer):
                            W[k][j]=update_kj_reg(W[k][j],err,hj[j],LR[q][l])
                                
                                
  
                    for j in range(hidden_layer):
                        for m in range(input_layer):
                            for k in range(output_layer):
                                V[j][m]=update_ji_reg(V[j][m],err,training_data.iloc[i],LR[q][l],netj[j],W[k][j],m,diff_active)
   
       
            print("Total Error:",error/(len(training_data)))
            err_q[q][l]=error/(len(training_data))
            W_q[q][l]=W
            V_q[q][l]=V
            print("Error:",err_q[q][l])
            print("The Weights Wkj",W_q[q][l])
            print("The Weights Vji",V_q[q][l])
            print("\n")
  


# In[128]:


##This is the main Regression Function
def regression_main(Number_of_Layers):
    print("\n The loss for the regression is Root mean Squared Loss\n")
    layers=Number_of_Layers
    if(layers==3):
        regression_layer_3(layers)
    else:
        print("\nThe number of layers you wnat to insert is completely incorrect")
        print("\n**Please Try again**")
        main_function()
        
        


# In[129]:


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


# In[126]:


main_function()


# In[72]:


#This is by using the Scikit_learn Library


# In[28]:


#Using scikit learn library
lin_reg = LinearRegression()
lin_reg.fit(training_data, y)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_


# In[29]:


train_predictions=lin_reg.predict(training_data)
print("\n The MSE Error for testing is:",mean_squared_error(y,train_predictions))

