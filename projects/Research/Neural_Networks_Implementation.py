#3-layers and 4-Layers Neural Network Implementation in python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# 3 LAYER NEURALNETWORK AND 4 LAYER NEURAL NETWORK


# In[3]:


#Neural Activation Functions
#1) Relu(Rectified Linear Unit)
#2) Sigmoid Activation Function
#3) Linear Activation Function
#4) Mean Squared error
#5) Sigmoid at output
#6) Softmax at teh output
#plan 1
#Define separate functions for each and every Activation Functions
#plan 2
# Define Separate functions for Linear Regression , Logistic Regression and Multi-Class Classification 


# In[4]:


##Number of Nodes in the hidden Layer
def hidden_layer_4():
    print("\n")
    hidden_layer_1=int(input("Enter the number of nodes in the Hidden Layer 1"))
    hidden_layer_2=int(input("Enter the number of nodes in the Hidden Layer 2"))
    return[hidden_layer_1,hidden_layer_2]


# In[5]:


##Loss Functions
def relu_output(value):
    if(value>=0):
        return(value)
    else:
        return(0)


# In[6]:


def linear_output(value):
    if(value>0):
        return(value)
    elif(value<0):
        return(value)
    else:
        return(0)


# In[7]:


#Sigmoid at the output of the logistic regression
def sigmoid_output(value):
    a=(1+math.exp(-value))
    return(1/a)


# In[8]:


#Rectified Linear Unit Activation Function
def relu_diff(value):
    if(value>=0):
        return(1)
    else:
        return(0)


# In[9]:


#Linear Activation Function
def linear_activation_diff(value):
    if(value>0):
        return(1);
    elif(value<0):
        return(-1)
    else:
        return 0


# In[10]:


#Sigmaoid Activation Function
def sigmoid_activation_diff(value):
    a=(1 + math.exp(-value))
    b=math.exp(-value)
    return(b/(a*a))


# In[11]:



# Creates a list containing 5 lists, each of 8 items, all set to 0
# w,h corresponds to width and height of the matrix
#w, h = 8, 5;
#a = [[0 for x in range(w)] for y in range(h)] ##This code is to create a 2-D array in python
## 0 for x in range (w) cirrespondsto number of elements in each row and(or the number of columns)
## for y in range(h) corresponds to the total number of rows
##for i in range(5):
  #  for j in range(8):
   #     a[i][j]=(input("Enter the number you want to insert"))
        
        


# In[12]:


##Change the Path Directory when ever you want
path=r"C:\Users\Pavan\Desktop\plastiq"
os.chdir(path)
print("Path Diectory:",os.getcwd())


# In[13]:


# Steps to follow about the data
## Load the data file into the training Environment  
## Make sure that the dataset is completely clean if it is not clean then deal with it
## After the data set is completely clean the calculate the total number of rows
##This calculated rows is used to train the model that many times to adapt with respect to each and every data point
names=["a","b","c","d"]
training_data=pd.read_csv("exam1.csv", header=None)
training_data  # This gives the total number of the Rows in the Data Set 



# In[14]:


y=training_data[3]
y


# In[15]:


del training_data[3]
training_data


# In[16]:


#Assuming that the data is completely clean 
# Now I am calculating the total numebr of times I had to loop
total_length=len(training_data)
total_length


# In[17]:


total_columns=len(training_data.columns)
total_columns


# In[18]:


#We can calculate the total number of input nodes directly
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
    


# In[19]:


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


# In[20]:


##Function to select the Activation Function
def select_activation(layers):
    print("Enter the Activation function for te Hidden Layer")
    print("relu for the Rectified Linear Unit")
    print("sigmoid_activation for Sigmoid Activation Unit")
    print("linear_activation for Linear Activation Function")
    if(layers==3):
        print("In the 3-layer Neural Network there will be only one Hidden Layer and the Activation Funvtion is Associated with that Function")
        activation_function=input("Enter the activation Function in the hidden Layer")
        return(activation_function)
    elif(layers==4):
        print("There are two Hidden Layers and each layer contains one Activation Function")
        activation_function_1=input("Enter the activation Function that you want to insert in Hidden layer 1")
        activation_function_2=input("Enter the activation Function taht you want to insert in Hidden Layer 2")
        return[activation_function_1,activation_function_2]


# In[21]:


def net_j(data_point,V,j,input_layer):
    sum_j=0
    for m in range(input_layer):
            sum_j=(sum_j)+(V[j][m]*(data_point[m]))
    return(sum_j)
    


# In[22]:


#This is just the Function of netj and Function F is called as the Activation Function
def values_hj(relu_output,netj):
    return(relu_output(netj))


# In[23]:


#Calculating the weights of the net k 
def net_k(hj,W,k,hidden_layer):
    sum_k=0
    for j in range(hidden_layer):
        sum_k=sum_k+(W[k][j]*(hj[j]))
    return(sum_k)


# In[24]:


## Updating the Wkj using the back propagation
def update_kj_reg(W,err,hj,LR):
    new_updated=W+(LR*(err)*hj)
    return(new_updated)


# In[25]:


##Updating the Vji using the bach propagation
def update_ji_reg(V,err,data_point,LR,netj,W,m,relu_diff):
    ji_updated=(V)+(LR*err*W*relu_diff(netj)*data_point[m])
    return(ji_updated)


# In[26]:


def update_kj_class(W,err,net,h,LR):
    new_updated=W+(LR*err*h*sigmoid_activation_diff(net))
    return(new_updated)


# In[27]:


def update_ji_class(V,err,data_point,LR,netj,netk,W,m,activation):
        ji_updated=(V)+(LR*err*sigmoid_activation_diff(netk)*W*relu_diff(netj)*data_point[m])
        return(ji_updated)


# In[28]:


# calculating the values lc i.e output of the hidden layer 1 
def values_lc_4(activation_1,netb):
    return(activation_1(netb))


# In[29]:


##I can convert all the functions to a single function taht will calculate the net
def net_b_4(data_point,V,b,input_layer):
    sum_b=0
    for m in range(input_layer):
        sum_b=(sum_b)+(V[b][m]*(data_point[m]))
    return(sum_b)


# In[30]:


def update_jb_reg_4(W,err,lc,LR,netj,P,b,activation_layer_2):
    W_updated=(W)+(LR*err*P*activation_layer_2(netj)*lc)
    return(W_updated)
    


# In[31]:


def update_bi_reg_4(V,LR,err,sum_update,data_point,m):
    bm_updated=(V)+(LR*err*sum_update*data_point[m])
    return(bm_updated)


# In[32]:


def test_model(W,V,y,training_data):
    output_layer=1                                                          #Because this is a Regression
    input_layer=input_layer_nodes(training_data,"NO")  
                                                    #Calculating the Total Number of Input Nodes
    #Need to initialize the hidden_layer_nodes_Function
    hidden_layer= 2 
    error=0
    err=0;
    
    for i in range(len(training_data)):
        ###I need to Initialize the Learning rate (Adaptive Learning rate)

        netj=[0 for m in range(hidden_layer)] ##Each node in the hidden layer has particulat net
        for j in range(hidden_layer):
            netj[j]=net_j(training_data.iloc[i],V,j,input_layer) 
       #calculating the values of hj
        hj=[0 for n in range(hidden_layer)]
        for j in range(hidden_layer):
            hj[j]=values_hj(relu_output, netj[j])
       #Now calculating the values of net k
        netk=[0 for i in range(output_layer)]
        for k in range(output_layer):
            netk=net_k(hj,W,hidden_layer,k)
       # This is the predicted Value
        predicted = netk
       #Till Now we calculated all the Values Now Find the error and back propagate
        err = (y[i]-predicted) #true label-Predicted value
        error=error+(err*err)
        print("Round",i)
        print("Error:",error)
    print("\n")
    print("Mean Squared Error:",(error/(len(training_data))))


# In[33]:


#3 LAYER NEURAL NETWORK FOR REGRESSION
def regression_layer_3(activation_function_1):
    output_layer=1                                                          #Because this is a Regression
    input_layer=input_layer_nodes(training_data,"NO")  
                                                    #Calculating the Total Number of Input Nodes
    #Need to initialize the hidden_layer_nodes_Function
    hidden_layer=int(input("Enter the number of nodes you want to insert in the Hidden Layer"))
  
    V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  #Initialize the weights from input layer to the hidden layer
    for j in range(hidden_layer):
        for m in range(input_layer):
            V[j][m]=np.random.uniform(0.001,0.001)
    W=[[0 for x in range(hidden_layer)] for y in range(output_layer)] #Intialize the weights from Hidden Layer to the output layer
    for k in range(output_layer):
        for j in range(hidden_layer):
            W[k][j]=np.random.uniform(0.001,0.0005)
    # Total Number of time you want to update 
    for number in range(10000):
        #Now Training and Updating
        print('\n')
        print("Number:",number)
        print("\n")
        error=0 #Initialize error to be zero
        err=0
        for i in range(total_length):
            netj=[0 for m in range(hidden_layer)] ##Each node in the hidden layer has particulat net
            for j in range(hidden_layer):
                netj[j]=net_j(training_data.iloc[i],V,j,input_layer) 
            #calculating the values of hj
            hj=[0 for n in range(hidden_layer)]
            for j in range(hidden_layer):
                hj[j]=values_hj(relu_output, netj[j])
            #Now calculating the values of net k
            netk=[0 for i in range(output_layer)]
            for k in range(output_layer):
                netk=net_k(hj,W,k,hidden_layer)
            # This is the predicted Value
            predicted = netk
            #Till Now we calculated all the Values Now Find the error and back propagate
            err = (y[i]-predicted) #true label-Predicted value
            error=error+(err*err)
            
        print("Error:",error)
        ##We calculated the error now we need to update the values based on taht error
       # if(error<5):
        ##    LR=0.0005
        #else:
        LR=0.00007
        ##Now we need to update the values based on the above Error
        ##First we need to update the top most most layer and send it back i.e Back Propagation
        
        for k in range(output_layer):
            for j in range(hidden_layer):
                W[k][j]=update_kj_reg(W[k][j],err,hj[j],LR)
        #By the above loop Wkj are updated
        #Now we need to update Vji i.e Hidden Layer to the input Layer
        activation=relu_diff
        for k in range(output_layer):
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=update_ji_reg(V[j][m],err,training_data.iloc[i],LR,netj[j],W[k][j],m,activation)
    print("The Weights Wkj",W)
    print("The Weights Vji",V)
    print("\n")
    print("Testing the model_now")
    test_model(W,V,y,training_data)


# In[64]:


#Regression Layer 4
def regression_layer_4(activation_1,activation_2):
    output_layer=1                                                          
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer=hidden_layer_4()
    hidden_layer_1=hidden_layer[0]
    hidden_layer_2=hidden_layer[1]                             
  
    V= [[0 for x in range(input_layer)] for y in range(hidden_layer_1)]  
    for b in range(hidden_layer_1):
        for m in range(input_layer):
            V[b][m]=np.random.uniform(0.01,0.05)
    W=[[0 for x in range(hidden_layer_1)] for y in range(hidden_layer_2)] 
    for j in range(hidden_layer_2):
        for b in range(hidden_layer_1):
            W[j][b]=np.random.uniform(0.001,0.005)
    P=[[0 for x in range(hidden_layer_2)]for y in range(output_layer)]
    for k in range(output_layer):
        for j in range(hidden_layer_2):
            P[k][j]=np.random.uniform(0.01,0.05)
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
                lc[c]=values_lc_4(relu_output,netb[c])
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
                hj[j]=values_hj(relu_output,netj[j])
            ##Now finally netK
            netk=[0 for i in range(output_layer)]
            for k in range(output_layer):
                netk=net_k(hj,P,k,hidden_layer_2)
            predicted = netk
            err = (y[i]-predicted)
            error=error+(err*err)
            
        print("Error:",error)
        ##We calculated the error now we need to update the values based on taht error
       # if(error<5):
        ##    LR=0.0005
        #else:
        LR=0.000001
        ##Now we need to update the values based on the above Error
        ##First we need to update the top most most layer and send it back i.e Back Propagation
        #updating Pkj
        for k in range(output_layer):
            for j in range(hidden_layer_2):
                P[k][j]=update_kj_reg(P[k][j],err,hj[j],LR)
        #By the above loop Wkj are updated
        #Now we need to update hidden_layer_2 to hidden_layer_1 Wjb
        for k in range(output_layer):
            for j in range(hidden_layer_2):
                for b in range(hidden_layer_1):
                    W[j][b]=update_jb_reg_4(W[j][b],err,lc[b],LR,netj[j],P[k][j],b,relu_diff)
        #Now we need to update from hidden_layer_1 to the input_layer
        sum_update=0;
        for b in range(hidden_layer_2):
            for m in range(input_layer):
                for k in range(output_layer):
                    for j in range(hidden_layer_2):
                        sum_update=sum_update+(P[k][j]*relu_diff(netj[j])*W[j][b]*relu_diff(netb[b]))
                
                V[b][m]= update_bi_reg_4(V[b][m],LR,err,sum_update,training_data.iloc[i],m)        
            
                                    
    print("The Weights Pkj:",P)
    print("The Weights Wjb:",W)
    print("The Weights Vbi:",V)
    print("\n")
    print("Testing the model_now")
                               
        
            


# In[65]:


##Classification of the 3-layer Neural Network
def classification_binary_3(activation_function):
    output_layer=1                                                      
    input_layer=input_layer_nodes(training_data,"NO")  
    hidden_layer= int(input("Enter the number of nodes you wnat to insert in the Hidden Layer"))             
    
    V= [[0 for x in range(input_layer)] for y in range(hidden_layer)]  ##Vji from the input to the hidden
    for j in range(hidden_layer):
        for m in range(input_layer):     #randint to declare a integer
            V[j][m]=np.random.randint(0,1) # random.uniform(to declare a float values)
            
    W=[[0 for x in range(hidden_layer)] for y in range(output_layer)]##Wkj the weights from the Hidden Layer to the output layer
    for k in range(output_layer):
        for j in range(hidden_layer):
            W[k][j]=np.random.randint(0,1)
            
    for number in range(100):##How many times you want to update the error
        print('\n')
        print("Number:",number)
        print("\n")
        error=0 
        err=0    
        ##Total number of Data points
        for i in range(total_length):   
            #calculating netj=vji*xi
            netj=[0 for m in range(hidden_layer)]
            for j in range(hidden_layer):
                netj[j]=net_j(training_data.iloc[i],V,j,input_layer) 
            # calculating the output from the hidden layer hj=F(netj)
            hj=[0 for n in range(hidden_layer)]
            for j in range(hidden_layer):
                hj[j]=values_hj(relu_output, netj[j])
            #Now calculating netk=Wkj*hj
            netk=[0 for i in range(output_layer)]
            for k in range(output_layer):
                netk=net_k(hj,W,hidden_layer,k)
            #In this logistic regression the netk has to go through sigmoid to get 0 and 1
            predicted = sigmoid_output(netk)
            err = (y[i]-predicted) 
            error=error+(err*err)

            
        print("Error:",error)
        #if(errorprev-error<=0.5): ##Converting this into a Adaptive learning rate
          #  LR=0.01
        #else:
        LR=0.000001
        ##Updating the values of Wkj
        for k in range(output_layer):
            for j in range(hidden_layer):
                W[k][j]=update_kj_class(W[k][j],err,netk,hj[j],LR)
        
        #Updating the values of Vkj   
        activation=relu_diff
        for k in range(output_layer):
            for j in range(hidden_layer):
                for m in range(input_layer):
                    V[j][m]=update_ji_class(V[j][m],err,training_data.iloc[i],LR,netj[j],netk,W[k][j],m,activation)
                    
    #Toatl Weights of Wkj
    print("The Weights Wkj",W)
    #Total Weights of Vji
    print("The Weights Vji",V)
    print("\n")
    


# In[66]:


##This is the main Regression Function
def regression_main(Number_of_Layers):
    layers=Number_of_Layers
    if(layers==3):
        activation_function=select_activation(layers)
        regression_layer_3(activation_function)
    elif(layers==4):
        print("In this 4-layer Neural Network there will be 2-Activation Functions and 2-Hidden Layers")
        ##In this select-activation should return multiple values and in python it returns as lists/tuple
        # Iam considering Lists because they are very easy to manipulate
        activation=select_activation(layers)
        activation_1=activation[0]
        activation_2=activation[1]
        regression_layer_4(activation_1,activation_2)
    else:
        print("The number of layers you wnat to insert is completely incorrect")
        print("**Please Try again**")
        print(main_function())
        
        


# In[67]:


#This is the main Classification Function(BINARY)
def classification_main(Number_of_Layers):
    layers=Number_of_Layers
    if(layers==3):
        activation_function=select_activation(layers)
        classification_binary_3(activation_function)
    elif(layers==4):
        activation=select_activation(layers)
        activation_1=activation[0]
        activation_2=activation[1]
        classification_binary_4(activation_1,activation_2)
    else:
        print("The number of layers you wnat to insert is completely incorrect")
        print("**Please Try again**")
        print(main_function())


# In[68]:


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



# In[69]:


main_function()


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(training_data, y)# Finding the parameters
lin_reg.intercept_, lin_reg.coef_


# In[ ]:


train_predictions=lin_reg.predict(training_data)
print("\n The MSE Error for testing is:",mean_squared_error(y,train_predictions))


# In[ ]:


a =np.random.rand(0.0,0.1)


# In[ ]:


a = np.random.uniform(0.0,0.2)
a

