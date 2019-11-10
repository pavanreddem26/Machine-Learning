
# coding: utf-8

# <h2>3-Layer Neural Network for Binary Classification using Squared Error and Cross Entropy Error

# In[110]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[111]:


##Relu Output---
def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[112]:


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[113]:


def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[114]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[115]:


def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[116]:


def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[117]:


def cross_entropy(value):
    a=math.exp(-value)
    return(a)


# In[118]:


##Change the Path Directory when ever you want--
path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[119]:


# Steps to follow about the data--
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 


# In[120]:


y=training_data[3]
y


# In[121]:


del training_data[3]
training_data


# In[122]:


#Assuming that the data is completely clean --
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[123]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[124]:


def select_activation(layers):
    
    print("Enter the Activation function for the Hidden Layer")
    print("Enter 'relu' for Rectified Linear Unit")
    print("Enter 'sigmoid' for Sigmoid Activation Unit")
    print("Enter 'linear' for Linear Activation Function")
    
    if(layers==3):
        print("In the 3-layer Neural Network there will be one Hidden Layer and one Activation Function")
        activation_function=input("Enter the activation Function in the hidden Layer")
        output_active=activ_function(activation_function)
        diff_active=diff_activation(activation_function)
        return[output_active,diff_active]
    else:
                
        print("You entered the wrong option")
        print('**Please Start Again**')
        main_function()


# In[125]:


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


# In[126]:


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


# In[127]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)


# In[128]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[129]:


def net_k(hj,W,k,hidden_layer):
    
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[130]:


def update_kj_class(W,err,net,h,LR):
    
    new_updated=W+(LR*err*h*sigmoid_diff(net))
    
    return(new_updated)


# In[131]:


#--
def update_ji_class(V,err,data_point,LR,netj,netk,W,m,diff_active):
    
        ji_updated=(V)+(LR*err*sigmoid_diff(netk)*W*diff_active(netj)*data_point[m])
        
        return(ji_updated)


# In[132]:


##
def update_kj_class_ce_3(W,p,netk,h,LR):
    
    W_updated=W+(LR*p*cross_entropy(netk)*h)
    
    return(W_updated)


# In[133]:


##
def update_ji_class_ce_3(V,p,data_point,LR,netj,netk,W,m,diff_activation):
    
    V_updated=V+(LR*p*cross_entropy(netk)*W*diff_activation(netj)*data_point[m])
    
    return(V_updated)


# In[134]:


def classification_binary_3_sq(layers):
    output_layer=1                                                      
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer= int(input("Enter the number of nodes you want to insert in the Hidden Layer"))  
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]    
    diff_active=hidden_activation[1]
    
    err_q=[[0 for i in range(5)]for j in range(5)]
    W_q=[[0 for i in range(5)]for j in range(5)]
    V_q=[[0 for i in range(5)]for j in range(5)]
    LR=[[0 for i in range(5)]for j in range(5)]
    
    for q in range(5):
    
                
        for l in range(5):
            
            print("\n")
            print("Round:",q)
            print("\n")
            
            V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
            W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] 
            for k in range(output_layer):
                for j in range(hidden_layer):
                    W[k][j]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
            
           
            LR[q][l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[q][l])
                
                
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
                        netk[k]=net_k(hj,W,hidden_layer,k)
                
                    for k in range(output_layer):
                        predicted = sigmoid_output(netk[k])
                        err =(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))

     
                    for k in range(output_layer):
                        for j in range(hidden_layer):
                            W[k][j]=update_kj_class(W[k][j],err,netk[k],hj[j],LR[q][l])
                                
                                
    
                    for j in range(hidden_layer):
                        for m in range(input_layer):
                            for k in range(output_layer):
                                V[j][m]=update_ji_class(V[j][m],err,training_data.iloc[i],LR[q][l],netj[j],netk[k],W[k][j],m,diff_active)
       
    
            err_q[q][l]=error/(len(training_data))
            W_q[q][l]=W
            V_q[q][l]=V
            print("Error:",err_q[q][l])
            print("The Weights Wkj",W_q[q][l])
            print("The Weights Vji",V_q[q][l])
            print("\n")


# In[135]:


### 3 layer neural network using Cross Entropy error
def classification_binary_3_ce(layers):
    output_layer=1                                                      
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer= int(input("Enter the number of nodes you wnat to insert in the Hidden Layer")) 
    
    print("\n")
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]
    diff_active=hidden_activation[1]
    
    err_q=[[0 for i in range(5)]for j in range(5)]
    W_q=[[0 for i in range(5)]for j in range(5)]
    V_q=[[0 for i in range(5)]for j in range(5)]
    LR=[[0 for i in range(5)]for j in range(5)]
    
    for q in range(5):
        
                
                
        for l in range(5):
            
            print("\n")
            print("Round:",q)

            
            V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
            W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] 
            for k in range(output_layer):
                for j in range(hidden_layer):
                    W[k][j]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
            
            LR[q][l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[q][l])
            print("\n")
            
            
            for number in range(1000):
                
                print("Number:",number)
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
                        netk[k]=net_k(hj,W,hidden_layer,k)
                        
                        
                    for k in range(output_layer):
                        predicted = sigmoid_output(netk[k])
                        p=(((predicted)**(y[i]))*((1-predicted)**y[i]))
                            
        
 
                    err = err+((-1)*(math.log(p,10)))  
            
       
                    for k in range(output_layer):
                        for j in range(hidden_layer):
                            W[k][j]=update_kj_class_ce_3(W[k][j],p,netk[k],hj[j],LR[q][l])

                    for j in range(hidden_layer):
                        for m in range(input_layer):
                            for k in range(output_layer):
                                V[j][m]=update_ji_class_ce_3(V[j][m],p,training_data.iloc[i],LR[q][l],netj[j],netk[k],W[k][j],m,diff_active)
                    
       
            err_q[q][l]=err
            W_q[q][l]=W
            V_q[q][l]=V
            print("Error:",err_q[q][l])
            print("The Weights Wkj",W_q[q][l])
            print("The Weights Vji",V_q[q][l])
            print("\n")


# In[136]:


def classification_main(Number_of_Layers):
    print("\n")
    print("\nPress 1 if you want to use Squared loss\n")
    print("\n press 2if you want to perform Cross Entropy Loss")
    loss=int(input("Enter the Loss that you want to consider\n"))
    layers=Number_of_Layers
    if(loss==1):
        if(layers==3):
            classification_binary_3_sq(layers)
        else:
            print("The number of layers you wnat to insert is completely incorrect")
            print("**Please Try again**")
            main_function()
    elif(loss==2):
        if(layers==3):
            classification_binary_3_ce(layers)
        else:
            print("The number of layers you wnat to insert is completely incorrect")
            print("**PLEASE TRY AGAIN**")
            main_function()
    else:
        print("The loss you entered doesnot exsist")
        print("**PLEASE TRY AGAIN**")
        main_function()


# In[137]:


##Main Function from which where I will call the other Functions to find the Necessary Coefficients
def main_function():
    print("Purpose is Classification_Binary \n")
    print("As of now the Number of Layers in neural Network is 3\n")
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Classification_Binary"):
            classification_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")
        print("**Please try again**")
        main_function()


# In[139]:


main_function()

