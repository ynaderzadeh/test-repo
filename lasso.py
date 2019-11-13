# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:42:19 2019

@author: ynaderzadeh
"""
# model is y = xw + noise

import numpy as np
import matplotlib.pyplot as plt
import copy

n = 500  #number of samples
k = 100  #features to be selected
d = 1000 #dimension of x


def w_gen(k,n):
    """ initilize weights according to question"""
    return np.array([i/k for i in range(1,k+1)]+[0 for i in \
                     range(n-k)]).reshape(n,1)

def err(y, x, w, lambda_):
    ss = np.sum(np.power(y - np.matmul(x, w),2),axis = 0)
    lasso_err = lambda_ * np.sum(np.abs(w),axis = 0)
    return ss[0] + lasso_err[0]

def lambda_max(x, y, d):
    y_center = y - np.mean(y)
    list_ = [2 * np.abs(np.matmul(x[:,i].T, y_center)) for i in range(d)]
    return max(list_)[0]

def lasso_solver(y, x, w_cur, d, lambda_, a,itr=100):
    
    it = 0
    while it < itr:
        it+=1
        print("iteration {} for lambda {}".format(it, lambda_))
        for dim in range(d):
            w_temp = np.concatenate((w_cur[:dim,:],w_cur[dim+1:,:]))
            #print(w_temp.shape)
            x_temp = np.concatenate((x[:,:dim],x[:,dim+1:]),axis=1)
            #print(x_temp.shape)
            ru = np.matmul((y - np.matmul(x_temp, w_temp)).T,\
                           x[:,dim].reshape((n,1)))[0]
            if ru < (-lambda_ / 2):
                w_cur[dim,0] = (ru + lambda_ / 2) / a[dim][0]
            elif (ru >= -lambda_ / 2) and (ru <= lambda_ / 2):
                w_cur[dim,0] = 0
            else:
                w_cur[dim,0] = (ru - lambda_ / 2) / a[dim][0]
    return w_cur

cov_m = np.identity(d)  #covariance of of x
mean_m = np.zeros(d)    # mean of x
np.random.seed(10)      #fix the seed

# generating x from multi_variabel normal distribution
x = np.random.multivariate_normal(mean_m, cov_m, n)
#print(x.shape)

#generationg noise
noise = np.random.normal(0, 1, (n,1))
#print(noise.shape)

# assiging w
w_data_generateed = w_gen(k,d)
#print(w_data_generateed.shape)

#generation y
y = np.matmul(x,w_data_generateed) + noise
#print(y.shape)

lambda_mx =  lambda_max(x, y, d)
print("Lambda max is {}".format(lambda_mx))

error = err(y, x, w_data_generateed, lambda_mx)
#print(error)

a = np.sum(np.power(x,2),axis = 0).reshape(d,1)

#initilizing the weights
w_cur = np.zeros((d,1)) 

lambda_ = lambda_mx   #initilize the first lambda
wight_lambda_log = {} # dictionary for saving weights for each lambda

while lambda_ > 0.01:
    # each weight vector is used for initial weight vector of next lambda
    w_cur = lasso_solver(y, x, w_cur, d, lambda_, a,15)
    print("err is {} for lambda {}".format(err(y, x, w_cur, lambda_),lambda_))
    
    # saving weights for each lambda
    wight_lambda_log[lambda_] = copy.deepcopy(w_cur)   
    lambda_ /= 1.5
    
lambdas = [[wight_lambda_log[i][j] for i in wight_lambda_log.keys()]\
            for j in range(d)]

plt.figure(1)
for i in range(d):
    
    plt.semilogx(list(wight_lambda_log.keys()), lambdas[i])
    plt.xlim(1300, 10**-1)
    
plt.xlabel("Lambda values")
plt.ylabel("Coeffitients")
plt.title("Regulization Path")
plt.savefig("regulization_path.pdf")


tpr = [np.sum(wight_lambda_log[i][:k,0] != 0) / k for i in \
       wight_lambda_log.keys()]
fdr = [0,0]+[np.sum(wight_lambda_log[i][k:,0] != 0)/ \
       np.sum(wight_lambda_log[i][:,0] != 0) for i in \
       wight_lambda_log.keys() if (i != (lambda_mx/1.5) and i != (lambda_mx))]

plt.figure(2)

plt.plot(fdr, tpr) 
plt.xlabel("FDR")
plt.ylabel("TPR")
plt.title("FDR-TPR for Various Lmanbda Values")
plt.savefig("FDR-TPR.pdf")


