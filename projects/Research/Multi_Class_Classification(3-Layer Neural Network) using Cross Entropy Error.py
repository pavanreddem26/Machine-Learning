
# coding: utf-8

# <h2> 3-Layer Neural network for Multi-Class Classification

# In[1]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[28]:


def soft_max(netk,output_layer,k):
    total_sum=0
    #calculating the denominator which is the sum of all the exponential functions
    for m in range(output_layer):
        total_sum=total_sum+(math.exp(netk[m]))
        
    return((math.exp(netk[k]))/total_sum)#returning the value for thar particular K


# In[29]:


def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[30]:


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[31]:


def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[32]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[33]:


def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[34]:


def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[35]:


def cross_entropy(value):
    a=math.exp(-value)
    return(a)


# In[36]:


path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[37]:


names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 


# In[38]:


y=training_data[3]
y


# In[39]:


del training_data[3]
training_data


# In[40]:


total_length=len(training_data)
total_length


# In[41]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[42]:


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


# In[43]:


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


# In[44]:


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


# In[45]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[46]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)


# In[47]:


def net_k_multi_class(hj,W,hidden_layer,k):
    
    net_k=0
    for j in range(hidden_layer):
        net_k=net_k+(W[k][j]*hj[j])
    return(net_k)


# In[48]:


def update_kj_multi_3(W,err,LR,hj):
    
    W_updated=(W)+(LR*err*hj)
    return(W_updated)


# In[49]:


##
def update_ji_multi_3(V,LR,netj,data_point,sum_ji_update,m,activation_function_1):
    
    V_updated=(V)+(LR*sum_ji_update*relu_diff(netj)*data_point[m])
    return(V_updated)


# In[54]:


def classification_multi_class_3(layers):
    output_layer=int(input("Enter the number of classes that you are predicting"))#number of classes==number of layers
    input_layer=input_layer_nodes(training_data,"NO") 
    hidden_layer=int(input("Enter the number of nodes you wnat to insert in the Hidden Layer"))
    
    print("\n")
    hidden_activation=select_activation(layers)
    output_active=hidden_activation[0]
    diff_active=hidden_activation[1]

    #  For one Hot Encoding
    si=[0 for i in range(output_layer)]
    for s in range(output_layer):
        print("Class:",s)
        si[s]=int(input("Enter the type of prediction of class :"))
    #Error 
    err_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[[0 for i in range(5)]for i in range(5)]
    
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
                    
            LR[l]=(1/(20**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
            for number in range(100):
                
                print('\n')
                print("Number:",number)
                print("\n")
                err=0    
                error=0
                
                for i in range(total_length):   
                    
                    netj=[0 for m in range(hidden_layer)]
                    for j in range(hidden_layer):
                        netj[j]=net_j(training_data.iloc[i],V,j,input_layer) 
            
                    hj=[0 for n in range(hidden_layer)]
                    for j in range(hidden_layer):
                        hj[j]=values_hj(output_active, netj[j])
                    netk=[0 for i in range(output_layer)]
                    for k in range(output_layer):
                        netk[k]=net_k_multi_class(hj,W,hidden_layer,k)
           
                    predicted=[0 for i in range(output_layer)]
                    error=[0 for i in range(output_layer)]
                    #use one hot Encoding to conert those values into Multi-Class which is suitable for ML algo
                    ##In the dataset how we are cconsidering those values i.e red is predicted as 1
            
                
                    #One Hot Encoding
                    yk=[0 for i in range(output_layer)]
                    for m in range(output_layer):
                        if(si[m]==y[i]):
                            for q in range(output_layer):
                                if(q==m):
                                    yk[q]=1
                                else:
                                    yk[q]=0;
                    
                
                    for k in range(output_layer):
                        predicted[k]=soft_max(netk[k],output_layer,k)
                        error[k]=(yk[k]-predicted[k])###Need to Change Error Function
                        print(error[k])
                    #Total Error

                    for k in range(output_layer):
                        err=err+error[k]
        
                    print("Error:",err)
   
        
                    #Till now we net values and calculated the error
                    #Now we need to updated the values in sucha a way that maximizes the likelihood
                    #Update Wkj
                    for k in range(output_layer):
                        for j in range(hidden_layer):
                            W[k][j]=update_kj_multi_3(W[k][j],error[k],LR[l],hj[j])
                    #Update Vji
                    sum_ji_update=0
                    for j in range(hidden_layer):
                        for m in range(input_layer):
                            for k in range(output_layer):
                                sum_ji_update=(sum_ji_update)+((y[k]-predicted[k])*W[k][j])
                        V[j][m]=update_ji_multi_3(V[j][m],LR[l],netj[j],training_data.iloc[i],sum_ji_update,m,diff_active)
        
    #Toatl Weights of Wkj
        err_q[q][l]=error
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q[q][l])
        print("The Weights Wkj",W_q[q][l])
        print("The Weights Vji",V_q[q][l])
        print("\n")
        print("Testing the model_now")


# In[55]:


def classification_multi_class_main(Number_of_Layers):
    layers=Number_of_Layers
    if(layers==3):
        classification_multi_class_3(layers)
    else:
        print("The number of layers you wnat to insert is completely incorrect")
        print("**Please Try again**")
        main_function()


# In[56]:


def main_function():
    print("Purpose is Classification_Multi_Class\n")
    print("As of now the Number of Layers in neural Network is 3\n")
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Classification_Multi_Class"):
          classification_multi_class_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")


# In[ ]:


main_function()

