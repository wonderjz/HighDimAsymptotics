#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:31:21 2024

@author: lijinze
"""


# Importing the NumPy library as np
import numpy as np

import matplotlib.pyplot as plt
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



def genxt(size,p,pzero,tau,seed,truepara):
    """
    3 factors model
    this function is to generating data in one period
    
    truepara is the true loadings
    """
    np.random.seed(seed)
    
    noise = np.random.normal(0,1)
    
    vectorinfo = []
    
    pzero = pzero
    #pzero = int(np.floor(min(p, 0.9*size))) # paper setting
    vectorno = np.random.normal(0,1,(max(0,(p-pzero)),1))
    ft0 = np.random.normal(0,1,(3,1)) # 3 factors
    ft = ft0
    #ft = np.append(ft0,1)  # add one constant term 
    
    '''
    for i in range(pzero):

        lambdai = np.random.normal(0,1,(3,1)) * (pzero**(-tau)) # 3-4
        noise = np.random.normal(0,1)
        xit = (np.transpose(lambdai) @ ft).item() + noise
        vectorinfo.append(xit) # vectorinfo is a list
    '''
    
    for i in range(p):
        if i < pzero:
            lambdai = np.random.normal(0,1,(3,1)) * (p**(-tau)) # 3-4
            noise = np.random.normal(0,1)
            xit = (np.transpose(lambdai) @ ft).item() + noise
            vectorinfo.append(xit) # vectorinfo is a list
        
            
    vectorinfo = np.reshape(vectorinfo,(len(vectorinfo),1 ) )
    predictor = np.concatenate([vectorinfo, vectorno])
    #vector = vectorinfo + vectorno
    
    noise2 = np.random.normal(0,1)
    y = (np.transpose(truepara) @ ft) + noise2 # true model
    
    return predictor,ft,y




def gentraindata(size,p,pzero,tau,truepara):
    x = []
    factor = []
    yvector = []
    
    for i in range(size):  #  i for seed
        seed = i
        predictort,factort,yt = genxt(size,p,pzero,tau,seed,truepara) 
        x.append(predictort)
        factor.append(factort)
        yvector.append(yt)
     
    xarray = np.stack(x,axis=0)
    xarray = np.squeeze(xarray) #dim:  50 *p
    yarray = np.stack(yvector,axis=0)
    
    return  xarray,yarray,factor


def gentestdata(size,p,pzero,tau,truepara):
    x = []
    factor = []
    yvector = []
    for i in range(size):  #  i for seed
        seed = 9999+i
        predictort,factort,yt = genxt(size,p,pzero,tau,seed,truepara) 
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


rlessmse_pinvlist = []
rlessr2_pinvlist = []


for thep in range(5,505,5):
    xarray,yarray,factor = gentraindata(size=100,p=thep,pzero=200,tau=1/2,truepara=[4,5,6])
    xarray_test,yarray_test,factor_test = gentestdata(size=50,p=thep,pzero=200,tau=1/2,truepara=[4,5,6])
    
    beta = np.linalg.pinv(xarray)@yarray
    y_pred = xarray_test@beta
    
    rlessmse_pinv = mse(yarray_test, y_pred)/np.shape(y_pred)[0] #divided by sample size
    rlessmse_pinvlist.append(rlessmse_pinv)
    rlessr2_pinv = r2_score(yarray_test, y_pred)/np.shape(y_pred)[0]
    rlessr2_pinvlist.append(rlessr2_pinv)
    print('i th:', thep)
    #print('【mse】: ', rlessmse_pinv, '【r2】: ', rlessr2_pinv)


### PCA

from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
pca = PCA(3) # 3 factors


pcamse_pinvlist = []
pcar2_pinvlist = []


for thep in range(5,505,5):
    xarray,yarray,factor = gentraindata(size=100,p=thep,pzero=200,tau=1/2,truepara=[4,5,6])
    xarray_test,yarray_test,factor_test = gentestdata(size=50,p=thep,pzero=200,tau=1/2,truepara=[4,5,6])
    # train

    # same with pca.fit
    #cov = np.cov(xarray)
    #(eva, evt) = np.linalg.eig(cov)
    '''
    pcamodel = pca.fit(xarray.T)
    newX = pcamodel.components_.T
    #newX = np.c_[ newX, np.ones(newX.shape[0]) ] #add constant term
    '''
    
    '''
    (eva, evt) = np.linalg.eig(xarray@np.transpose(xarray))
    #eva3 = eva[0:3]
    newX = evt[:, 0:3] * np.sqrt(len(evt))
    '''

    # pca fit
    #newX = pca.fit_transform(xarray/np.sqrt(len(xarray)))
    newX = pca.fit_transform(xarray)
    #eigenvalues = pca.explained_variance_
    #print(eigenvalues)


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
    print('i th:', thep)
    #print('【mse】: ', mse_pinv, '【r2】: ', r2_pinv)
   


# plot the whole
plt.plot(pcamse_pinvlist[0:],'k',rlessmse_pinvlist[0:],'b')
plt.ylim(0,6)
plt.legend(['pca','ridgeless'],)
plt.axvline(x =100/5, color = 'k', linestyle='dashed')# n=100
plt.axvline(x =200/5, color = 'r', linestyle='dashed')# pzero

plt.show()




#plot    
plt.plot(rlessmse_pinvlist)
#plt.ylim(0,6)
plt.show()

plt.plot(pcamse_pinvlist)
#plt.ylim(0,6)
plt.show()


# round the number then plot
ridgeless_list = [ round(elem, 3)  for elem in rlessmse_pinvlist]
pca_list = [ round(elem, 3)  for elem in pcamse_pinvlist]
plt.plot(pca_list[0:],'k',ridgeless_list[0:],'b')
plt.ylim(0,6)
plt.legend(['pca','ridgeless'],)
plt.axvline(x =100, color = 'k', linestyle='dashed')
plt.axvline(x =200, color = 'r', linestyle='dashed')
plt.show()




# plot skip small numbers
plt.plot(pcamse_pinvlist[300:],'k',rlessmse_pinvlist[300:],'b')
plt.ylim(0,5)
plt.legend(['pca','ridgeless'],)

plt.show()

