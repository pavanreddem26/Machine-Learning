
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


def soft_max(netk,output_layer,k):
    total_sum=0
    #calculating the denominator which is the sum of all the exponential functions
    for m in range(output_layer):
        total_sum=total_sum+(math.exp(netk[m]))
        
    return((math.exp(netk[k]))/total_sum)#returning the value for thar particular K


# In[3]:


def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[4]:


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[5]:


def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[6]:


def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[7]:


def linear_diff(value):
    if(value>0):
        return(1)
    elif(value<0):
        return(-1)
    else:
        return (0)


# In[8]:


def sigmoid_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[9]:


def cross_entropy(value):
    a=math.exp(-value)
    return(a)


# In[10]:


path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[11]:


names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 


# In[12]:


y=training_data[3]
y


# In[13]:


del training_data[3]
training_data


# In[14]:


total_length=len(training_data)
total_length


# In[15]:


def hidden_layer_4():
    
    
    print("\n")
    hidden_layer_1=int(input("Enter the number of nodes in the Hidden Layer 1"))
    hidden_layer_2=int(input("Enter the number of nodes in the Hidden Layer 2"))
    return[hidden_layer_1,hidden_layer_2]##In python you can return two values but in C we can't return two values


# In[16]:


def input_layer_nodes(training_data,Dimension_reduction):
    
    
    if(Dimension_reduction=="NO"):
        input_nodes=len(training_data.columns)
        if(input_nodes==0):
            return(1)   #Adding this as the bias node
        else:
            return(input_nodes) #Make sure to remove the Label(Y) from the training data
                                #If you don't want to remove the label(Y) then consider input_nodes-1
    


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


def update_kj_multi_3(W,err,LR,hj):
    
    W_updated=(W)+(LR*err*hj)
    return(W_updated)


# In[21]:


def update_jb_multi_4(W,LR,netj,data_point,sum_jb_update,b,activation_function_1):
    
    W_updated=(W)+(LR*sum_jb_update*relu_diff(netj)*data_point[b])
    return(W_updated)


# In[22]:


def update_bi_multi_4(V,LR,sum_bi_update,data_point,i):
    
    V_updated=(V)+(LR*sum_bi_update*data_point[i])
    return(V_updated)


# In[23]:


def net_b_4(data_point,V,b,input_layer):
    
    sum_b=0
    for m in range(input_layer):
        sum_b=(sum_b)+(V[b][m]*(data_point[m]))
    return(sum_b)


# In[24]:


def values_lc_4(output_active_1,netb):
    
    return(output_active_1(netb))


# In[25]:


def net_j(data_point,V,j,input_layer):
    
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)


# In[26]:


def values_hj(output_active,netj):
    
    return(output_active(netj))


# In[27]:


def net_k_multi_class(hj,W,hidden_layer,k):
    
    net_k=0
    for j in range(hidden_layer):
        net_k=net_k+(W[k][j]*hj[j])
    return(net_k)


# In[32]:


def classification_multi_class_4(layers):
    output_layer=int(input("Enter the number of classes that you are predicting"))#number of classes==number of layers
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
    
    #  For one Hot Encoding
    si=[0 for i in range(output_layer)]
    for s in range(output_layer):
        print("Class:",s)
        si[s]=int(input("Enter the type of prediction of class :"))

    
    
    err_q=[[0 for i in range(5)]for i in range(5)]
    P_q=[[0 for i in range(5)]for i in range(5)]
    W_q=[[0 for i in range(5)]for i in range(5)]
    V_q=[[0 for i in range(5)]for i in range(5)]
    LR=[[0 for i in range(5)] for i in range(5)]
    
    
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
                    
                    
            LR[q][l]=(1/(10**(l+1)))
            print("\n")
            print("Round of LR",(l+1))
            print(LR[l])
            print("\n")
            
                
            for number in range(10000):
                
                
                print('\n')
                print("Number:",number)
                print("\n")
                error=0 
                err=0
                
                for i in range(total_length):
                    
                    
                    netb=[0 for m in range(hidden_layer_1)]
                    #calculating the values to netb=sum(vbi*xi)
                    for b in range(hidden_layer_1):
                        netb[b]=net_b_4(training_data.iloc[i],V,b,input_layer) 
                    #calculating the values of lc
                    lc=[0 for n in range(hidden_layer_1)]
                    for c in range(hidden_layer_1):
                        lc[c]=values_lc_4(output_active_1,netb[c])
                    #calculating the netj
                     #Initializing the netj
                    netj=[0 for i in range(hidden_layer_2)]
                    #Calculating all the values of netj
                   # We are using the same function as we used for Regression of 3 layers because the function looks 
                   #that is changing is the values that we pass into them
                    for j in range(hidden_layer_2):
                        netj[j]=net_j(lc,W,j,hidden_layer_1)
                    #After calculating the net
                    hj=[0 for n in range(hidden_layer_2)]
                    for j in range(hidden_layer_2):
                        hj[j]=values_hj(output_active_2,netj[j])
                    ##Now finally netK sice this is a multi-class we will have many outputs at the output
                    netk=[0 for i in range(output_layer)]
                    for k in range(output_layer):
                        netk[k]=net_k_multi_class(hj,P,hidden_layer_2,k)
                    predicted=[0 for i in range(output_layer)]
                    error=[0 for i in range(output_layer)]
                  
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
                        predicted[k]=soft_max(netk,output_layer,k)
                        error[k]=(yk[k]-predicted[k])
                    #Total Error
                    for k in range(output_layer):
                        err=err+error[k]
         
     
                    ##Now we need to update the values based on the above Error
                    ##First we need to update the top most most layer and send it back i.e Back Propagation
                    #updating Pkj
                    for k in range(output_layer):
                        for j in range(hidden_layer_2):
                            P[k][j]=update_kj_multi_3(P[k][j],error[k],LR[q][l],hj[j])

                    sum_jb_update=0
                    for j in range(hidden_layer_2):
                        for b in range(hidden_layer_1):
                            for k in range(output_layer):
                                sum_jb_update=(sum_jb_update)+((error[k])*P[k][j])
                        
                            W[j][b]=update_jb_multi_4(W[j][b],LR[q][l],netj[j],lc,sum_jb_update,b,diff_active_2)
       
    
                    sum_bi_update=0;
                    for b in range(hidden_layer_2):
                        for i in range(input_layer):
                            for k in range(output_layer):
                                for j in range(hidden_layer_2):
                                    sum_bi_update=sum_bi_update+((error[k]*P[k][j]*diff_active_2(netj[j])*W[j][b]*diff_active_1(netb[b])))
                 
                            V[b][i]= update_bi_multi_4(V[b][i],LR[q][l],sum_bi_update,training_data.iloc[i],i)        
            
                                    
        err_q[q][l]=error
        P_q[q][l]=P
        W_q[q][l]=W
        V_q[q][l]=V
        print("Error:",err_q)
        print("The Weights Pkj",P_q)
        print("The Weights Wjb",W_q)
        print("The Weights Vbi",V_q)
        print("\n")


# In[33]:


def classification_multi_class_main(Number_of_Layers):
    layers=Number_of_Layers
    if(layers==4):
        classification_multi_class_4(layers)
    else:
        print("The number of layers you wnat to insert is completely incorrect")
        print("**Please Try again**")
        main_function()


# In[34]:


def main_function():
    print("Purpose is Classification_Multi_Class\n")
    print("As of now the Number of Layers in neural Network is 4\n")
    Purpose_of_Neural_Network = str(input("Enter the purpose of your Neural network:")) 
    Number_of_Layers=int(input("Enter the number of Layers you want in your Neural Network:"))
    if(Purpose_of_Neural_Network == "Classification_Multi_Class"):
          classification_multi_class_main(Number_of_Layers)
    else:
        print("Entered Choice Does Not Exsist\n")


# In[35]:


main_function()

