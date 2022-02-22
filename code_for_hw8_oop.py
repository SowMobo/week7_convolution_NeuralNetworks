# This OPTIONAL problem has you extend your homework 7
# implementation for building neural networks.  
# PLEASE COPY IN YOUR CODE FROM HOMEWORK 7 TO COMPLEMENT THE CLASSES GIVEN HERE

# Recall that your implementation from homework 7 included the following classes:
    # Module, Linear, Tanh, ReLU, SoftMax, NLL and Sequential

######################################################################
# OPTIONAL: Problem 2A) - Mini-batch GD
######################################################################

from ast import Mod
from turtle import forward
from matplotlib.pyplot import axis, cla

#####################################################################
# Linear module
#####################################################################

class Module:
    def step(self, lrate): pass  # For modules w/o weights
    
# Each Linear module has a forward method that takes in a batch of activations A (from previous layer) and
# returns a batch of pre-activations Z.
#
#
# Each Linear module has a backward method that takes in dLdZ and returns dLdA. This module also compute and stores dLdW 
# and dLdW0, the gradient with respect to the wights.

class Linear(Module):
    def __init__(self, m, n):
        self.m, self.n = (m, n) #(in size, out size)
        self.W0 = np.zeros([self.n, 1]) # (nx1)
        self.W = np.random.normal(0, 1.0*m**(-.5), [m, n]) # (m, n)
        
    def forward(self, A): # think A0 as X
        self.A = A  #(m, b) Hint: make sure you understand what b stands for
        return (self.W.T @ self.A) + self.W0 # return Z of size (n, m) x (m, b) + (n, 1) = (n, b)
    
    def backward(self, dLdZ): # dLdZ is (n, b), uses self.A
        self.dLdW = self.A @ dLdZ.T
        self.dLdW0 = dLdZ @ np.ones((dLdZ.shape[1], 1)) # May be changed if an error occurred in the expression 
        return self.W @ dLdZ
        
    def step(self, lrate): # gradient descent step updates the weights
        self.W -= lrate*self.dLdW
        self.W0 -= lrate*self.dLdW0

##################################################################
# Activation module
##################################################################
# Each activation module has a forward that takes in a batch of pre-activations Z and returns a batch of activations A.
#
# Each activation has a backward that takes in dLdA and returns dLdZ, with the exception of softmax, where we assume dLdZ passed in

class Tanh(Module): # Layer activation
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    
    def backward(self, dLdA):
        return dLdA * (1.0 - (self.A ** 2))

class ReLU(Module): # Layer activation
    def forward(self, Z):
        self.A = np.maximum(Z, 0) # A is (n, b)
        return self.A
    
    def backward(self, dLdA):
        return dLdA * (self.A != 0)

class SoftMax(Module):
    def forward(self, Z):
        den = np.sum(np.exp(Z), axis=0) # (1, b)
        return np.exp(Z)/den # (n, b)
    
    def backward(self, dLdZ): # Assume that dLdZ is passed in
        return dLdZ
    
    def class_fun(self, Ypred): # returns class indices
        return np.argmax(Ypred, axis=0) # (1, b)

###################################################################
# Loss Module 
###################################################################
# Each loss module has a forward method takes in a batch of predictions Ypred (from previous layer) and labels Y and 
# returns a scalar loss value.
#
#The NLL module has a backward method that returns dLdZ, the gradient with respect to the preactivation to SoftMax
# (not the activation), since we are always pairing SoftMax activation with NLL loss

class NLL(Module):  # Loss
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return -np.sum(Y*np.log(Ypred)) # returns a scalar loss value
    
    def backward(self): # use stored self.Ypred, self.Y
        return (self.Ypred - self.Y)    # (n, b)
            
#############################################################################
# Neural Networks Implementation
#############################################################################        

class Sequential:
    def __init__(self, modules, loss):            
        self.modules = modules
        self.loss = loss

    def mini_gd(self, X, Y, iters, lrate, notif_each=None, K=10):
        D, N = X.shape

        np.random.seed(0)
        num_updates = 0
        indices = np.arange(N)
        while num_updates < iters:

            np.random.shuffle(indices)
            X = X[:, indices]  # Your code
            Y = Y[:, indices]  # Your code

            for j in range(m.floor(N/K)):
                if num_updates >= iters: break

                # Implement the main part of mini_gd here
                # Your code
                Xt = X[:, j*K:(j+1)*K]
                Yt = Y[:, j*K:(j+1)*K]
                
                Ypred = self.forward(Xt)
                loss = self.loss.forward(Ypred, Yt)
                err = self.loss.backward()
                self.backward(err)
                self.step(lrate)
                num_updates += 1
                

    def forward(self, Xt):                        
        for m in self.modules: Xt = m.forward(Xt)
        return Xt

    def backward(self, delta):                   
        for m in self.modules[::-1]: delta = m.backward(delta)

    def step(self, lrate):    
        for m in self.modules: m.step(lrate)
          
######################################################################
# OPTIONAL: Problem 2B) - BatchNorm
######################################################################

class BatchNorm(Module):    
    def __init__(self, m):
        np.random.seed(0)
        self.eps = 1e-20
        self.m = m  # number of input channels
        
        # Init learned shifts and scaling factors
        self.B = np.zeros([self.m, 1])
        self.G = np.random.normal(0, 1.0 * self.m ** (-.5), [self.m, 1])
        
    # Works on m x b matrices of m input channels and b different inputs
    def forward(self, A):# A is m x K: m input channels and mini-batch size K
        # Store last inputs and K for next backward() call
        self.A = A
        self.K = A.shape[1]
        
        self.mus = np.mean(self.A, axis=1, keepdims=True)  # Your Code
        self.vars = np.var(self.A, axis=1, keepdims=True)  # Your Code

        # Normalize inputs using their mean and standard deviation
        self.norm = (self.A - self.mus)/(np.sqrt(self.vars) + self.eps)  # Your Code
            
        # Return scaled and shifted versions of self.norm
        return self.G * self.norm + self.B # Your Code

    def backward(self, dLdZ):
        # Re-usable constants
        std_inv = 1/np.sqrt(self.vars+self.eps)
        A_min_mu = self.A-self.mus
        
        dLdnorm = dLdZ * self.G
        dLdVar = np.sum(dLdnorm * A_min_mu * -0.5 * std_inv**3, axis=1, keepdims=True)
        dLdMu = np.sum(dLdnorm*(-std_inv), axis=1, keepdims=True) + dLdVar * (-2/self.K) * np.sum(A_min_mu, axis=1, keepdims=True)
        dLdX = (dLdnorm * std_inv) + (dLdVar * (2/self.K) * A_min_mu) + (dLdMu/self.K)
        
        self.dLdB = np.sum(dLdZ, axis=1, keepdims=True)
        self.dLdG = np.sum(dLdZ * self.norm, axis=1, keepdims=True)
        return dLdX

    def step(self, lrate):
        self.B -= lrate*self.dLdB  # Your Code
        self.G -= lrate*self.dLdG  # Your Code
        return


######################################################################
# Tests
######################################################################
def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)
  
def for_softmax(y):
    return np.vstack([1-y, y])
  
# For problem 1.1: builds a simple model and trains it for 3 iters on a simple dataset
# Verifies the final weights of the model
def mini_gd_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 3, lrate=0.005, K=1)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]
  
# For problem 1.2: builds a simple model with a BatchNorm layer
# Trains it for 1 iter on a simple dataset and verifies, for the BatchNorm module (in order): 
# The final shifts and scaling factors (self.B and self.G)
# The final running means and variances (self.mus_r and self.vars_r)
# The final 'self.norm' value
def batch_norm_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), BatchNorm(2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.mini_gd(X,Y, iters = 1, lrate=0.005, K=2)
    return [np.vstack([nn.modules[3].B, nn.modules[3].G]).tolist(), \
    np.vstack([nn.modules[3].mus_r, nn.modules[3].vars_r]).tolist(), nn.modules[3].norm.tolist()]