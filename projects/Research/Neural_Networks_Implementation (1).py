
# coding: utf-8

# In[ ]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


# 3 LAYER NEURALNETWORK AND 4 LAYER NEURAL NETWORK


# In[ ]:


#Neural Activation Functions
#1) Relu(Rectified Linear Unit)
#2) Sigmoid Activation Function
#3) Linear Activation Function
#4) Mean Squared error
#5) Sigmoid at output
#6) Softmax at the output
#plan 1
#Define separate functions for each and every Activation Functions
#plan 2
# Define Separate functions for Linear Regression , Logistic Regression and Multi-Class Classification 


# <h2> Defining all the Activation Functions and their Differentials

# In[ ]:


#--
def soft_max(netk,output_layer,k):
    total_sum=0
    #calculating the denominator which is the sum of all the exponential functions
    for m in range(output_layer):
        total_sum=total_sum+(math.exp(netk[m]))
        
    return((math.exp(netk[k]))/total_sum)#returning the value for thar particular K
        


# In[ ]:


##Relu Output---
def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[ ]:


#Linear output--
def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(-value)
    else:
        return(0)


# In[ ]:


#Cross Entropy--
def cross_entropy(value):
    a=math.exp(-value)
    return(a)


# In[ ]:


#Sigmoid at the output of the logistic regression--
def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[ ]:


#Rectified Linear Unit Activation Function--
def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[ ]:


#Linear Activation Function--
def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[ ]:


#Sigmaoid Activation Function--
def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[ ]:



# Creates a list containing 5 lists, each of 8 items, all set to 0
# w,h corresponds to width and height of the matrix
#w, h = 8, 5;
#a = [[0 for x in range(w)] for y in range(h)] ##This code is to create a 2-D array in python
## 0 for x in range (w) cirrespondsto number of elements in each row and(or the number of columns)
## for y in range(h) corresponds to the total number of rows
##for i in range(5):
  #  for j in range(8):
   #     a[i][j]=(input("Enter the number you want to insert"))
        
        


# In[ ]:


##Change the Path Directory when ever you want--
path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[ ]:


# Steps to follow about the data--
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 



# In[ ]:


y=training_data[3]
y


# In[ ]:


del training_data[3]
training_data


# In[ ]:


#Assuming that the data is completely clean --
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[ ]:


total_columns=len(training_data.columns)
total_columns


# In[ ]:


#When the Neural Network is four layered than this function will be called--
def hidden_layer_4():
    
    
    print("\n")
    hidden_layer_1=int(input("Enter the number of nodes in the Hidden Layer 1"))
    hidden_layer_2=int(input("Enter the number of nodes in the Hidden Layer 2"))
    return[hidden_layer_1,hidden_layer_2]##In python you can return two values but in C we can't return two values


# In[ ]:


#We can calculate the total number of input nodes directly--
# We you want to perform(PCA) which is Dimension Reduction then you need to calculate the total number of features after
#performing the PCA
#If you are NOT performing the the Dimension Reduction then you can Calculate Directly from the training data set
#For all the layers the number of Input Nodes
def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[ ]:


#COMMENTS THAT EXPALIN THE BELOW CODE PROPERLY
#inially we are intiliazing all the values to be some random values and in each step we will adjust the 
#values of the weights or parametes in such a way that mininmuzes and maximizes the error
#first we need to intialize all the w,s to some random value
#First Declare and Initialize them and then Update them
#==================================================================================
# First intitialize the Vji with some random values use 2-D array to define the weights
#Intializing the 2-D array 
#V[j][i] i.e the weights from the layer i to j
# First one is the width and the second one is the height
#input layer corresponds to the total of columns in the MATRIX
#Hidden layer cooresponds to the total Number of rows in the Matrix
#====================================================================================
#    #By Now we intialized all the weights with some Random Values
    #Now what left is to Calculate the output, error and update the weightes according to that weights
    #  Weights_j- Function that calculates the total netj
    ## outer loop which is looping over all the data points


# In[ ]:


##Function to select the Activation Function--
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
    
    elif(layers==4):
        
        print("There are two Hidden Layers and each layer contains one Activation Function")
        activation_function_1=input("Enter the activation Function that you want to insert in Hidden layer 1")
        output_active_1=activ_function(activation_function_1)
        activation_function_2=input("Enter the activation Function that you want to insert in Hidden Layer 2")
        output_active_2=activ_function(activation_function_2)
        diff_active_1=diff_activation(activation_function_1)
        diff_active_2=diff_activation(activation_function_2)
        
        return[output_active_1,output_active_2,diff_active_1,diff_active_2]


# In[ ]:


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


# In[ ]:


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


# <h2> All the Net functions and the Intermediate Values 

# In[ ]:


#--
def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)
    


# In[ ]:


#This is just the Function of netj and Function F is called as the Activation Function--
def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[ ]:


#Calculating the weights of the net k --
def net_k(hj,W,k,hidden_layer):
    
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[ ]:


## Updating the Wkj using the back propagation--
def update_kj_reg(W,err,hj,LR):
    
    new_updated=W+(LR*(err)*hj)
    return(new_updated)


# In[ ]:


##Updating the Vji using the bach propagation--
def update_ji_reg(V,err,data_point,LR,netj,W,m,diff_active):
    
    ji_updated=(V)+(LR*err*W*diff_active(netj)*data_point[m])
    return(ji_updated)


# In[ ]:


#--
def update_kj_class(W,err,net,h,LR):
    
    new_updated=W+(LR*err*h*sigmoid_activation_diff(net))
    return(new_updated)


# In[ ]:


#--
def update_ji_class(V,err,data_point,LR,netj,netk,W,m,activation):
    
        ji_updated=(V)+(LR*err*sigmoid_activation_diff(netk)*W*relu_diff(netj)*data_point[m])
        return(ji_updated)


# In[ ]:


#--
def values_lc_4(output_active_1,netb):
    
    return(output_active_1(netb))


# In[ ]:


##--
def net_b_4(data_point,V,b,input_layer):
    
    sum_b=0
    for m in range(input_layer):
        sum_b=(sum_b)+(V[b][m]*(data_point[m]))
    return(sum_b)


# In[ ]:


##
def update_jb_reg_4(W,err,lc,LR,netj,P,b,diff_active_2):
    
    W_updated=(W)+(LR*err*P*diff_active_2(netj)*lc)
    return(W_updated)
    


# In[ ]:


##
def update_bi_reg_4(V,LR,err,sum_update,data_point,m):
    
    bm_updated=(V)+(LR*err*sum_update*data_point[m])
    return(bm_updated)


# In[ ]:


##
def update_kj_class_ce_3(W,p,netk,h,LR):
    
    W_updated=W+(LR*p*cross_entropy(netk)*h)
    return(W_updated)


# In[ ]:


##
def updated_ji_class_ce(V,p,data_point,LR,netj,netk,W,m,diff_activation):
    
    V_updated=V+(LR*p*cross_entropy(netk)*W*diff_activation(netj)*data_point[m])


# In[ ]:


##
def update_jb_class_4_sq(W,err,lc,LR,netj,P,relu_diff,netk):
    
    w_updated=(W)+(LR*err*sigmoid_activation_diff(netk)*P*relu_diff(netj)*lc)
    return(w_updated)


# In[ ]:


##
def update_kj_class_4_sq(P,err,netk,hj,LR):
    
    p_updated=(p)+(LR*err*sigmoid_activation_diff(netk)*hj)
    return(p_updated)


# In[ ]:


##
def update_bi_class_4_sq(V,LR,err,sum_update,data_point,netk,m):
    
    v_updated=(v)+(LR*err*sigmoid_activation_diff(netk)*sum_update*data_point[m])


# In[ ]:


##
def update_jb_class_4_ce(W,p,lc,LR,netj,netk,P,relu_diff):
    
    W_updated=W+(LR*p*cross_entropy(netk)*P*relu_diff(netj)*lc)
    return(W_updated)


# In[ ]:


##
def update_bi_class_4_ce(V,LR,p,sum_update,data_point,netk,m):
    
    V_updated=(V)+(LR*p*cross_entropy(netk)*sum_update*data_point[m])
    return(V_updated)


# In[ ]:


##
def update_jb_multi_4(W,LR,netj,data_point,sum_jb_update,b,activation_function_1):
    
    W_updated=(W)+(LR*sum_jb_update*relu_diff(netj)*data_point[b])
    return(W_updated)


# In[ ]:


##
def update_bi_multi_4(V,LR,sum_bi_update,data_point,i):
    
    V_updated=(V)+(LR*sum_bi_update*data_point[i])
    return(V_updated)


# In[ ]:


##
def net_k_multi_class(hj,W,hidden_layer,k):
    
    net_k=0
    for j in range(hidden_layer):
        net_k=net_k+(W[k][j]*hj[j])
    return(net_k)


# In[ ]:


##
def update_kj_multi_3(W,err,LR,hj):
    
    W_updated=(W)+(LR*err*hj)
    return(W_updated)


# In[ ]:


##
def update_ji_multi_3(V,LR,netj,data_point,sum_ji_update,m,activation_function_1):
    
    V_updated=(V)+(LR*sum_ji_update*relu_diff(netj)*data_point[m])
    return(V_updated)






# <h2> 3--Layer Neural Network for Regression:

# In[ ]:


#3 LAYER NEURAL NETWORK FOR REGRESSION--
def regression_layer_3(layers):
    output_layer=1                                                          
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
    print("\n")
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]
    diff_active=hidden_activation[1]
    #Instead Of Using the one fixed weights i am changing the weights and choosing the one that has minimum error
    err_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
    for q in range(5):
        
        #This is for setting 5 learning rates for each and every weights performed
        
        for l in range(5):
            
            
            print("\n")
            print("Round:",q)
            print("\n")
            V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
            W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
            for k in range(output_layer):
                for j in range(hidden_layer):
                    W[k][j]=np.random.uniform((-1)*(1/(10**(q+1))),1/((10)**(q+1)))
                    
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            # Total Number of time you want to update 
            
            
            for number in range(100):
                
                print('\n')
                error=0 
                print("Number:",number)
                err=0
                
                
                for i in range(total_length):
                    
                    netj=[0 for m in range(hidden_layer)] ##Each node in the hidden layer has particulat net
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
                        err =err+(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))
            
                print("Total Error:",error)

        
                for k in range(output_layer):
                    for j in range(hidden_layer):
                        W[k][j]=update_kj_reg(W[k][j],err,hj[j],LR[l])
                                
  
                for j in range(hidden_layer):
                    for m in range(input_layer):
                        for k in range(output_layer):
                            V[j][m]=update_ji_reg(V[j][m],err,training_data.iloc[i],LR[l],netj[j],W[k][j],m,diff_active)
   

        err_q[q][l]=error
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Wkj",W_q)
        print("The Weights Vji",V_q)
        print("\n")
        print("Testing the model_now")


# <h2> 4 -Layer Neural Network for Regression:

# In[ ]:


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
    
    err_q=[[0 for i in range(5)]for i in range(5)]
    P_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
    for q in range(5):

    
        
        for l in range(5):
            
            
            print("\n")
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
            
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
            for number in range(100):  ##Need to take care of the error
                
                print("Number:",number)
                print("\n")
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
                        err =err+(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))
            
            
                print("Error:",error)
                for k in range(output_layer):
                    for j in range(hidden_layer_2):
                        P[k][j]=update_kj_reg(P[k][j],err,hj[j],LR[l])
                
                for j in range(hidden_layer_2):
                    for b in range(hidden_layer_1):
                        for k in range(output_layer):
                            W[j][b]=update_jb_reg_4(W[j][b],err,lc[b],LR[l],netj[j],P[k][j],b,diff_active_2)
            
                
                #Now we need to update from hidden_layer_1 to the input_layer
                sum_update=0
                for b in range(hidden_layer_2):
                    for m in range(input_layer):
                        for k in range(output_layer):
                            for j in range(hidden_layer_2):
                                sum_update=sum_update+(P[k][j]*diff_active_2(netj[j])*W[j][b]*diff_active_1(netb[b]))
        
                        V[b][m]= update_bi_reg_4(V[b][m],LR[l],err,sum_update,training_data.iloc[i],m) 
                
    
        err_q[q][l]=error
        P_q[q][l]=P
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Pkj",P_q)
        print("The Weights Wjb",W_q)
        print("The Weights Vbi",V_q)
        print("\n")    


# <h2> 3 Layer Neural Network for Binary_Classification using Squared Loss as Error:

# In[ ]:


##Classification of the 3-layer Neural Network--
def classification_binary_3_sq(layers):
    output_layer=1                                                      
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer= int(input("Enter the number of nodes you want to insert in the Hidden Layer"))  
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]    
    diff_active=hidden_activation[1]
    
    err_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
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
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
                
                
            for number in range(100):
                
                
                print('\n')
                print("Number:",number)
                print("\n")
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
                        predicted = sigmoid_output(netk[0])
                        err =err+(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))

            
                  
                print("Error:",error)
        
                for k in range(output_layer):
                    for j in range(hidden_layer):
                        W[k][j]=update_kj_class(W[k][j],err,netk,hj[j],LR[l])
    
                for j in range(hidden_layer):
                    for m in range(input_layer):
                        for k in range(output_layer):
                            V[j][m]=update_ji_class(V[j][m],err,training_data.iloc[i],LR[l],netj[j],netk,W[k][j],m,diff_active)
       
    
        err_q[q][l]=error
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q[q][l])
        print("The Weights Wkj",W_q[q][l])
        print("The Weights Vji",V_q[q][l])
        print("\n")
        print("Testing the model_now")
    


# <h2> 4-Layer Neural Network for Binary_Classification using Squared Error:

# In[ ]:


##Layer-4 classification_binary using Square loss error
def classification_binary_4_sq(layers):
    output_layer=1                                                          
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer=hidden_layer_4()
    hidden_layer_1=hidden_layer[0]
    hidden_layer_2=hidden_layer[1]  
    print("\n")
    hidden_activation=select_activation(layers)
    output_active_1=hidden_activation[0]
    output_active_2=hidden_activation[1]
    diff_active_1=hidden_activation[2]
    diff_active_2=hidden_activation[3]
    
    err_q=[[0 for i in range(5)]for i in range(5)]
    P_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
    
    for q in range(5):

        
        for l in range(5):
            
            print("\n")
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
            
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
            
            for number in range(100):
                
                
                print('\n')
                print("Number:",number)
                print("\n")
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
                        predicted = sigmoid_activation_diff(netk[k])
                        err =err+(y[i]-predicted)
                        error=error+((y[i]-predicted)*(y[i]-predicted))
                                
            
                print("Error:",error)

                for k in range(output_layer):
                    for j in range(hidden_layer_2):
                        P[k][j]=update_kj_reg(P[k][j],err,netk,hj[j],LR)
       
                for j in range(hidden_layer_2):
                    for b in range(hidden_layer_1):
                        for k in range(output_layer):
                            W[j][b]=update_jb_reg_4(W[j][b],err,lc[b],LR,netj[j],P[k][j],diff_active_2,netk)
        
                sum_update=0
                for b in range(hidden_layer_2):
                    for m in range(input_layer):
                        for k in range(output_layer):
                            for j in range(hidden_layer_2):
                                sum_update=sum_update+(P[k][j]*diff_active_2(netj[j])*W[j][b]*diff_active_1(netb[b]))
                
                        V[b][m]= update_bi_reg_4(V[b][m],LR,err,sum_update,training_data.iloc[i],netk,m)      
                    
        err_q[q][l]=error
        P_q[q][l]=P
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Pkj",P_q)
        print("The Weights Wjb",W_q)
        print("The Weights Vbi",V_q)
        print("\n") 
                               


# <h2> 3-Layer Neural Network for Binary_classification using Cross Entropy as Error:

# In[ ]:


### 3 layer neural network using Cross Entropy error
def classification_binary_3_ce(layers):
    output_layer=1                                                      
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer= int(input("Enter the number of nodes you wnat to insert in the Hidden Layer")) 
    
    print("\n")
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]
    diff_active=hidden_activation[1]
    
    err_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
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
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
            
            for number in range(1000):
                
                
                print('\n')
                print("Number:",number)
                print("\n")
                p=1;
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
                        predicted = sigmoid_output(netk[0])
                        p=p*(((predicted)**(y[i]))*((1-predicted)**y[i]))
        
 
                err = err+(-1)*(math.log(p,10))    
                print("Error:",err)
       
                for k in range(output_layer):
                    for j in range(hidden_layer):
                        W[k][j]=update_kj_class_ce_3(W[k][j],p,netk,hj[j],LR[l])

                for j in range(hidden_layer):
                    for m in range(iput_layer):
                        for k in range(output_layer):
                            V[j][m]=update_ji_class_ce(V[j][m],p,training_data.iloc[i],LR[l],netj[j],netk,W[k][j],m,diff_active)
                    
       
        err_q[q][l]=error
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Wkj",W_q)
        print("The Weights Vji",V_q)
        print("\n")
        print("Testing the model_now")
    


# <h2> 4-layer Neural Network for Binary_Classification using Cross Entropy Error:

# In[ ]:


# 4-Layer Neural network using Cross Entropy Error
def classification_binary_4_ce(layers):
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

    
    err_q=[[0 for i in range(5)]for i in range(5)]
    P_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[0 for i in range(5)]
    
    
    for q in range(5):
        
       
        
        for l in range(5):
            
            print("\n")
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
    
            
            
            LR[l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
            for number in range(10000):
                
                print('\n')
                print("Number:",number)
                print("\n")
                p=1 
                err=0
                
                
                for i in range(total_length):
                    
                    
                    netb=[0 for m in range(hidden_layer_1)]
                    for b in range(hidden_layer_1):
                        netb[b]=net_b_4(training_data.iloc[i],V,b,input_layer) 
            
                    lc=[0 for n in range(hidden_layer_1)]
                    for c in range(hidden_layer_1):
                        lc[c]=values_lc_4(output_active_1,netb[c])
      
                    netj=[0 for i in range(hidden_layer_2)]
                    for j in range(hidden_layer_2):
                        netj[j]=net_j(lc,W,j,hidden_layer_1)
    
                    hj=[0 for n in range(hidden_layer_2)]
                    for j in range(hidden_layer_2):
                        hj[j]=values_hj(output_active_2,netj[j])
           
                    netk=[0 for i in range(output_layer)]
                    for k in range(output_layer):
                        netk=net_k(hj,P,k,hidden_layer_2)
                
                    for k in range(output_layer):
                        predicted = sigmoid_output(netk[k])
                        p=p*(((predicted)**(y[i]))*((1-predicted)**y[i]))
        

            err = err+(-1)*(math.log(p,10))  
            
            print("Error:",err)

            for k in range(output_layer):
                for j in range(hidden_layer_2):
                    P[k][j]=update_kj_class_ce_3(P[k][j],p,netk,hj[j],LR)
        
            for j in range(hidden_layer_2):
                for b in range(hidden_layer_1):
                    for k in range(output_layer):
                        W[j][b]=update_jb_class_4_ce(W[j][b],p,lc[b],LR,netj[j],netk,P[k][j],diff_active_2)
        
        
            sum_update=0;
            for b in range(hidden_layer_2):
                for m in range(input_layer):
                    for k in range(output_layer):
                        for j in range(hidden_layer_2):
                            sum_update=sum_update+(P[k][j]*diff_active_2(netj[j])*W[j][b]*diff_active_1(netb[b]))
                
                    V[b][m]= update_bi_class_4_ce(V[b][m],LR,p,sum_update,training_data.iloc[i],netk,m)        
            
                                    
        err_q[q][l]=error
        P_q[q][l]=P
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Pkj",P_q)
        print("The Weights Wjb",W_q)
        print("The Weights Vbi",V_q)
        print("\n")


# <h3> Regression Main Function:

# In[ ]:


##This is the main Regression Function
def regression_main(Number_of_Layers):
    print("\n The loss for the regression is Root mean Squared Loss\n")
    layers=Number_of_Layers
    if(layers==3):
        regression_layer_3(layers)
    elif(layers==4):
        regression_layer_4(layers)
    else:
        print("\nThe number of layers you wnat to insert is completely incorrect")
        print("\n**Please Try again**")
        main_function()
        
        


# <h3>Classification(Binary) Main Function:

# In[ ]:


#This is the main Classification Function(BINARY)
def classification_main(Number_of_Layers):
    print("\n")
    print("\nPress 1 if you want to use Squared loss\n")
    print("\n press 2if you want to perform Cross Entropy Loss")
    loss=int(input("Enter the Loss that you want to consider\n"))
    layers=Number_of_Layers
    if(loss==1):
        if(layers==3):
            classification_binary_3_sq(layers)
        elif(layers==4):
            classification_binary_4_sq(layers)
        else:
            print("The number of layers you wnat to insert is completely incorrect")
            print("**Please Try again**")
            main_function()
    elif(loss==2):
        if(layers==3):
            classification_binary_3_ce(layers)
        elif(layers==4):

            classification_binary_4_ce(layers)
        else:
            print("The number of layers you wnat to insert is completely incorrect")
            print("**PLEASE TRY AGAIN**")
            main_function()
    else:
        print("The loss you entered doesnot exsist")
        print("**PLEASE TRY AGAIN**")
        main_function()


# <h3> Classification(MULTI- CLASS) Main Function:

# In[ ]:


def classification_multi_class_main(Number_of_Layers):
    layers=Number_of_Layers
    if(layers==3):
        classification_multi_class_3(layers)
    elif(layers==4):
        classification_multi_class_4(layers)
    else:
        print("The number of layers you wnat to insert is completely incorrect")
        print("**Please Try again**")
        main_function()
    


# In[ ]:


##Main Function from which where I will call the other Functions to find the Necessary Coefficients
def main_function():
    print("Purpose is either Classification_Binary Or Regression Or Classification_Multi_Class\n")
    print("As of now the Number of Layers in neural Network is either 3 or 4 and later I will genralize it to the hole\n")
#The entered input is always considered as string and to change that we need to convert that into 
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Regression" ):
            regression_main(Number_of_Layers)
    elif(Purpose_of_Neural_Network == "Classification_Binary"):
            classification_main(Number_of_Layers)
    elif(Purpose_of_Neural_Network == "Classification_Multi_Class"):
          classification_multi_class_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")



# In[ ]:


main_function()


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(training_data, y)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_


# In[ ]:


train_predictions=lin_reg.predict(training_data)
print("\n The MSE Error for testing is:",mean_squared_error(y,train_predictions))


