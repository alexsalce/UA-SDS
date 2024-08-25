# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 06:51:10 2024

@author: Alex Salce
"""


import numpy as np
import numpy.linalg as la

class gdDataSARAH():
    def __init__(self, A, b, outer_loop_iter, inner_loop_iter, batch_size, epsilon=None):
        self.A = A
        self.b = b
        self.outer_iter = outer_loop_iter  
        self.inner_iter = inner_loop_iter
        self.batch = batch_size
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
    def LipLS(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        # L = 2 * la.norm(A)**2
        L=0
        for i in range(len(A)):
            Li = 2 * la.norm(A[i,:])**2
            if Li>L:
                L = Li
        return L
    def muLS(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        hess = 2*A.T @ A
        mineig = min(np.linalg.eig(hess)[0])
        return mineig
    def objective_function_ls(self,x,A=None,b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        return np.linalg.norm(A@x - b)**2
    def sGradLS(self, x, indices, A=None, b=None,batch_size=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        x = x
        batch = self.batch if batch_size is None else batch_size
        i = indices
        Abatch = A[i, :]
        bbatch = b[i]
        # grad = self.n * self.GradLS(x,Abatch,bbatch)
        grad = self.GradLS(x,Abatch,bbatch)
        return grad
    def GradLS(self, x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        # grad = (2 * self.n / len(b)) *	A.T @ (A  @ x - b)
        grad = 2 *	A.T @ (A  @ x - b)
        return grad
    


class gdDataADAM():
    def __init__(self, A, b, maxit, alpha, beta1, beta2, batch_size, epsilon):
        self.A = A
        self.b = b
        self.maxit = maxit        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.maxit = maxit
        self.batch = batch_size
        self.epsilon = epsilon
        self.n = self.A.shape[0]
        self.nfeat = self.A.shape[1]
        self.Ab = np.c_[A.reshape(self.n, -1), b.reshape(self.n, 1)]
        
    def LipLS(self, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        L = 2 * la.norm(A)**2
        eta = 1/L
        return eta
    def objective_function_ls(self,x,A=None,b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        return np.linalg.norm(A@x - b)**2
    def sGradLS(self, x, indices, A=None, b=None,batch_size=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        x = x
        batch = self.batch if batch_size is None else batch_size
        i = indices
        Abatch = A[i, :]
        bbatch = b[i]
        grad = self.GradLS(x,Abatch,bbatch)
        return grad
    def GradLS(self, x, A=None, b=None):
        A = self.A if A is None else A
        b = self.b if b is None else b
        grad = (2 * self.n / len(b)) *	A.T @ (A  @ x - b)
        return grad
    
    
def objective_function(x,A,b):
    return np.linalg.norm(A@x - b)**2