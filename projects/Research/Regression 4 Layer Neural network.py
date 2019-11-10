
# coding: utf-8

# <h2> 4-Layer Neural network for Linear Regression

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


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[4]:


def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[5]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[6]:


def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[7]:


def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[8]:


##Change the Path Directory when ever you want--
path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[9]:


# Steps to follow about the data--
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 


# In[10]:


y=training_data[3]
y


# In[11]:


del training_data[3]
training_data


# In[12]:


#Assuming that the data is completely clean --
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[13]:


total_columns=len(training_data.columns)
total_columns


# In[14]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[15]:


def hidden_layer_4():
    
    
    print("\n")
    hidden_layer_1=int(input("Enter the number of nodes in the Hidden Layer 1"))
    hidden_layer_2=int(input("Enter the number of nodes in the Hidden Layer 2"))
    return[hidden_layer_1,hidden_layer_2]##In python you can return two values but in C we can't return two values


# In[16]:


##Function to select the Activation Function--
def select_activation(layers):
    
    print("Enter the Activation function for the Hidden Layer")
    print("Enter 'relu' for Rectified Linear Unit")
    print("Enter 'sigmoid' for Sigmoid Activation Unit")
    print("Enter 'linear' for Linear Activation Function")
    if(layers==4):
        print("There are two Hidden Layers and each layer contains one Activation Function")
        activation_function_1=input("Enter the activation Function that you want to insert in Hidden layer 1")
        output_active_1=activ_function(activation_function_1)
        activation_function_2=input("Enter the activation Function that you want to insert in Hidden Layer 2")
        output_active_2=activ_function(activation_function_2)
        diff_active_1=diff_activation(activation_function_1)
        diff_active_2=diff_activation(activation_function_2)
        return[output_active_1,output_active_2,diff_active_1,diff_active_2]
    else:
        print("**You Entered the wrong option\n")
        print("**Please start Again**")
        main_function()


# In[17]:


#Selecting activation function--
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


# In[18]:


## Selecting differential Activation Functions
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


# In[19]:


def update_kj_reg_4(W,err,hj,LR):
    
    new_updated=W+(LR*(err)*hj)
    return(new_updated)


# In[20]:


def update_jb_reg_4(W,err,lc,LR,netj,P,b,diff_active_2):
    
    W_updated=(W)+(LR*err*P*diff_active_2(netj)*lc)
    return(W_updated)


# In[21]:


##
def update_bi_reg_4(V,LR,err,sum_update,data_point,m):
    
    bm_updated=(V)+(LR*err*sum_update*data_point[m])
    return(bm_updated)


# In[22]:


def net_b_4(data_point,V,b,input_layer):
    
    sum_b=0
    for m in range(input_layer):
        sum_b=(sum_b)+(V[b][m]*(data_point[m]))
    return(sum_b)


# In[23]:


def values_lc_4(output_active_1,netb):
    
    return(output_active_1(netb))


# In[24]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)
    


# In[25]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[26]:


def net_k(hj,W,k,hidden_layer):
    
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[33]:


#Regression Layer 4
def regression_layer_4(layers):
    output_layer=1                                                          
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer=hidden_layer_4()
    hidden_layer_1=hidden_layer[0]
    hidden_layer_2=hidden_layer[1]   
    hidden_activation=select_activation(layers)
    output_active_1=hidden_activation[0]
    output_active_2=hidden_activation[1]
    diff_active_1=hidden_activation[2]
    diff_active_2=hidden_activation[3]
    
    err_q=[[0 for i in range(5)]for j in range(5)]
    P_q=[[0 for i in range(5)]for j in range(5)]
    W_q=[[0 for i in range(5)]for j in range(5)]
    V_q=[[0 for i in range(5)]for j in range(5)]
    LR=[[0 for i in range(5)]for j in range(5)]
    
    for q in range(5):

    
        
        for l in range(5):
            
            print("Round:",q)
            print("\n")
            V= [[0 for x in range(input_layer)] for y in range(hidden_layer_1)]
            for b in range(hidden_layer_1):
                for m in range(input_layer):
                    V[b][m]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
            
            W=[[0 for x in range(hidden_layer_1)] for y in range(hidden_layer_2)] 
            for j in range(hidden_layer_2):
                for b in range(hidden_layer_1):
                    W[j][b]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
            P=[[0 for x in range(hidden_layer_2)]for y in range(output_layer)]
            for k in range(output_layer):
                for j in range(hidden_layer_2):
                    P[k][j]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
            
            
            LR[q][l]=(1/(10**(l+1)))
            print("Round of LR",(l+1))
            print(LR[q][l])
            
            for number in range(100):  ##Need to take care of the error
                
                error=0 
                err=0
                
                for i in range(total_length):
                    
                    netb=[0 for m in range(hidden_layer_1)]
                    for b in range(hidden_layer_1):
                        netb[b]=net_b_4(training_data.iloc[i],V,b,input_layer) 
                        
                    lc=[0 for n in range(hidden_layer_1)]
                    for b in range(hidden_layer_1):
                        lc[b]=values_lc_4(output_active_1,netb[b])
                        
                    netj=[0 for i in range(hidden_layer_2)]
                    for j in range(hidden_layer_2):
                        netj[j]=net_j(lc,W,j,hidden_layer_1)
                        
                    hj=[0 for n in range(hidden_layer_2)]
                    for j in range(hidden_layer_2):
                        hj[j]=values_hj(output_active_2,netj[j])
                        
                    netk=[0 for i in range(output_layer)]
                    for k in range(output_layer):
                        netk[k]=net_k(hj,P,k,hidden_layer_2)
                        
                    for k in range(output_layer):
                        predicted = netk[k]
                        err =(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))
            
            
                    for k in range(output_layer):
                        for j in range(hidden_layer_2):
                            P[k][j]=update_kj_reg_4(P[k][j],err,hj[j],LR[q][l])
                
                    for j in range(hidden_layer_2):
                        for b in range(hidden_layer_1):
                            for k in range(output_layer):
                                W[j][b]=update_jb_reg_4(W[j][b],err,lc[b],LR[q][l],netj[j],P[k][j],b,diff_active_2)
            
                
   
                    sum_update=0
                    for b in range(hidden_layer_2):
                        for m in range(input_layer):
                            for k in range(output_layer):
                                for j in range(hidden_layer_2):
                                    sum_update=sum_update+(P[k][j]*diff_active_2(netj[j])*W[j][b]*diff_active_1(netb[b]))
        
                            V[b][m]= update_bi_reg_4(V[b][m],LR[q][l],err,sum_update,training_data.iloc[i],m) 
                

            err_q[q][l]=error/(len(training_data))
            P_q[q][l]=P
            W_q[q][l]=W
            V_q[q][l]=V
            print("Error:",err_q[q][l])
            print("The Weights Pkj",P_q[q][l])
            print("The Weights Wjb",W_q[q][l])
            print("The Weights Vbi",V_q[q][l])


# In[34]:


##This is the main Regression Function
def regression_main(Number_of_Layers):
    print("\n The loss for the regression is Root mean Squared Loss\n")
    layers=Number_of_Layers
    if(layers==4):
        regression_layer_4(layers)
    else:
        print("\nThe number of layers you wnat to insert is completely incorrect")
        print("\n**Please Try again**")
        main_function()
        


# In[35]:


def main_function():
    print("Purpose is Regression \n")
    print("As of now the Number of Layers in neural Network is 4\n")
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Regression" ):
            regression_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")
        print("**Please Start Again**")
        main_function()


# In[36]:


main_function()


# In[37]:


#Using scikit learn library
lin_reg = LinearRegression()
lin_reg.fit(training_data, y)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_


# In[38]:


train_predictions=lin_reg.predict(training_data)
print("\n The MSE Error for testing is:",mean_squared_error(y,train_predictions))

