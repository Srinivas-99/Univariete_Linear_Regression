# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:30:13 2021

@author: srini
"""

import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

'''

Univariate Linear Regression model with gradient decent optimizing method

'''

np.random.seed(2)

#getting data

height= np.random.uniform(low=150,high=190,size=(20,1))
height=np.sort(height,axis=0)
#print (height,np.shape(height))
weight= 0.37*height+12.89
#print(weight,np.shape(weight))

#Adding noise to the clean data so that we can get real time values 

noise= np.random.rand(weight.shape[0],np.shape(weight)[1])
weight=weight+noise
#print(weight,np.shape(weight))

#ploting the data 
fig=plt.figure()
plt.scatter(height,weight)
plt.xlabel("height")
plt.ylabel("weight")
plt.show()

'''
Univariate linear Regression model has the form

y=0o+01*x

where y = weight  (dependent variable)
      x = height  (independent variable) 
'''

#Scaling the height feature (normalizing the feature)


height_=height
#print(height)


#Standard Scaling


height_mu=np.mean(height)
height_sig=np.std(height)
height=(height-height_mu)/(height_sig+1e-6)
#print(height)
 
# for train the data changing the variable names 

x_ones=np.ones_like(height)
#print(x_ones)
x_true=np.concatenate((x_ones,height),axis=1)
#print(x_true,np.shape(x_true))
y_true=weight

def loss_fun(y_cal,y_true):
    loss=np.mean((y_cal-y_true)**2)
    return loss
def grad(y_cal,y_true,x):
    ar0=np.mean(y_cal-y_true)
    ar1=np.mean((y_cal-y_true)*x)
    return [ar0,ar1]

def testing(x_true,y_true,para,alpha):
    y_cal=(para*x_true).sum(axis=1)
    y_cal=np.reshape(y_cal,(y_cal.shape[0],1))
    #finding loss
    loss=loss_fun(y_cal,y_true)
    #finding gradients
    gra=grad(y_cal,y_true,x_true)
    #updating para
    new_para=para-(alpha*np.array(gra))
    
    return loss,new_para

#number of trianing iterrations
iter=50
alpha=0.5 #learning rate
losses=[]

#initializing the both parameters to one
para=np.ones((1,2),dtype=np.float32)
#print(para)

for i in range(iter):
    loss,new_para=testing(x_true,y_true,para,alpha)
    losses.append(loss)
    para=new_para
'''
Ploting the loss factor
'''
iters=np.arange(1,iter+1)
graph=plt.figure()
plt.plot(iters,losses)
plt.xlabel("iterations")
plt.ylabel("Training loss")
plt.title("loss factor plot")
plt.show()

'''
Ploting the reslut model with defined data
'''
x_demo=np.arange(150,190)
x_demo=(x_demo-height_mu)/(height_sig+1e-6)
y_demo=para[:,0]+para[:,1]*x_demo
fig=plt.figure()
plt.scatter(height,weight,label="testing data")
plt.plot(x_demo,y_demo,label="our model",c="r")
plt.xlabel("Height(cm)")
plt.ylabel("Weight(Kg)")
plt.legend()
plt.show()


'''
predicting the weight for input data

'''

inp_height=int(input("Enter the height:\n "))
inp_high=(inp_height-height_mu)/(height_sig+1e-6)
out_weight=para[:,0]+para[:,1]*inp_high
print("weight of a person with height %d is %.3f"%(inp_height,out_weight[0]))

#ploting the output for input height in the graph


fig=plt.figure()
plt.scatter(height,weight,c='b')
plt.scatter(inp_high,out_weight,c="r")
plt.xlabel("Height(cm)")
plt.ylabel("Weight(Kg)")
plt.title("output result")
plt.show()

