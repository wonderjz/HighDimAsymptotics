#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:10:54 2024

@author: lijinze
"""

# Importing the NumPy library as np
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
# Generating a 1D array of p0 random numbers between 0 and 1 using np.random.random()


#pzero = 500 
#p = 1000
#tau = 1/4
#truepara = [30,40,50]

#np.random.normal(mu, sigma, (n_sample, n_param))



def genxt(size,p,tau,seed,truepara):
    """
    3 factors model
    this function is to generating data in one period
    """
    np.random.seed(seed)
    
    noise = np.random.normal(0,1)
    
    vectorinfo = []
    pzero = int(np.floor(min(p, 0.9*size))) # paper setting
    vectorno = np.random.normal(0,1,(max(0,(p-pzero)),1))
    ft0 = np.random.normal(0,1,(3,1)) # 3 factors
    ft = np.append(ft0,1)  # add one constant term factor
    
    for i in range(pzero):

        lambdai = np.random.normal(0,1,(4,1)) * (pzero**(-tau))
        noise = np.random.normal(0,1)
        xit = (np.transpose(lambdai) @ ft).item() + noise
        vectorinfo.append(xit) # vectorinfo is a list
        
    vectorinfo = np.reshape(vectorinfo,(len(vectorinfo),1 ) )
    predictor = np.concatenate([vectorinfo, vectorno])
    #vector = vectorinfo + vectorno
    
    noise2 = np.random.normal(0,1)
    y = (np.transpose(truepara) @ ft) + noise2 # true model
    
    return predictor,ft,y




def gentraindata(size,p,tau,truepara):
    x = []
    factor = []
    yvector = []
    
    for i in range(size):  #  i for seed
        seed = i
        predictort,factort,yt = genxt(size,p,tau,seed,truepara) 
        x.append(predictort)
        factor.append(factort)
        yvector.append(yt)
     
    xarray = np.stack(x,axis=0)
    xarray = np.squeeze(xarray) #dim:  50 *p
    yarray = np.stack(yvector,axis=0)
    
    return  xarray,yarray,factor


def gentestdata(size,p,tau,truepara):
    x = []
    factor = []
    yvector = []
    for i in range(size):  #  i for seed
        seed = 9999+i
        predictort,factort,yt = genxt(size,p,tau,seed,truepara) 
        x.append(predictort)
        factor.append(factort)
        yvector.append(yt)
     
    xarray = np.stack(x,axis=0)
    xarray = np.squeeze(xarray) #dim:  50 *p
    yarray = np.stack(yvector,axis=0)
    
    return  xarray,yarray,factor




### lasso
xarray,yarray,factor = gentraindata(size=100,pzero=500,p=1000,tau=1/2,truepara=[30,40,50])
xarray_test,yarray_test,factor_test = gentestdata(size=50,pzero=500,p=1000,tau=1/2,truepara=[30,40,50])

lassomodel = LassoCV(cv=10, random_state=0,tol=1e-2).fit(xarray, yarray)
#lassomodel = Lasso(alpha=10,max_iter=10000)
#lassomodel.fit(predictorarray,yarray)

#coef = lassomodel.coef_
#print(lassomodel.coef_)
#print(lassomodel.intercept_)

  
y_pred = lassomodel.predict(xarray_test)
y_pred = y_pred.reshape((-1,1)) 
mse_lasso = mse(yarray_test, y_pred)/np.shape(y_pred)[0] #divided by sample size
r2_lasso = r2_score(yarray_test, y_pred)/np.shape(y_pred)[0]
print('【mse】: ', mse_lasso, '【r2】: ', r2_lasso)




### ridgeless regression
#beta = np.linalg.pinv(xarray)@yarray
#y_pred = xarray_test@beta


mse_pinvlist = []
r2_pinvlist = []


for thep in range(5,1000,5):
    xarray,yarray,factor = gentraindata(size=100,p=thep,tau=1/2,truepara=[1,2,3,4])
    xarray_test,yarray_test,factor_test = gentestdata(size=50,p=thep,tau=1/2,truepara=[1,2,3,4])
    
    beta = np.linalg.pinv(xarray)@yarray
    y_pred = xarray_test@beta
    
    mse_pinv = mse(yarray_test, y_pred)/np.shape(y_pred)[0] #divided by sample size
    mse_pinvlist.append(mse_pinv)
    r2_pinv = r2_score(yarray_test, y_pred)/np.shape(y_pred)[0]
    r2_pinvlist.append(r2_pinv)
    print('【mse】: ', mse_pinv, '【r2】: ', r2_pinv)


### PCA

from sklearn.decomposition import PCA

pca = PCA(3) # 3 factors


pcamse_pinvlist = []
pcar2_pinvlist = []


for thep in range(5,1000,5):
    xarray,yarray,factor = gentraindata(size=100,p=thep,tau=1/2,truepara=[1,2,3,4])
    xarray_test,yarray_test,factor_test = gentestdata(size=50,p=thep,tau=1/2,truepara=[1,2,3,4])
    # train
    newX = pca.fit_transform(xarray)
    #newX = np.c_[ newX, np.ones(newX.shape[0]) ] #add constant term

    olsregr = LinearRegression()
    olsregr.fit(newX, yarray)

    # Make predictions 
    newX_test = pca.fit_transform(xarray_test) # add constant
    #newX_test = np.c_[ newX_test, np.ones(newX_test.shape[0]) ] 
    y_pred = olsregr.predict(newX_test)
    
    mse_pinv = mse(yarray_test, y_pred)/np.shape(y_pred)[0] #divided by sample size
    pcamse_pinvlist.append(mse_pinv)
    r2_pinv = r2_score(yarray_test, y_pred)/np.shape(y_pred)[0]
    pcar2_pinvlist.append(r2_pinv)
    print('【mse】: ', mse_pinv, '【r2】: ', r2_pinv)
   
    
plt.plot(pcamse_pinvlist)

   
plt.plot(pcamse_pinvlist[100:],'g',mse_pinvlist[100:],'r')
plt.show()
