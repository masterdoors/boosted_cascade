'''
Created on 18 окт. 2023 г.

@author: keen
'''
import warnings
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS, inplace_identity_derivative,inplace_tanh_derivative, inplace_logistic_derivative, inplace_relu_derivative    
from sklearn.model_selection import train_test_split
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.base import (
    _fit_context, is_classifier)

from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy

from sklearn.exceptions import ConvergenceWarning 
from sklearn.utils import (
    _safe_indexing,
    gen_batches,
    shuffle,
    check_random_state)

import scipy
import copy
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np
from itertools import chain

from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

import nlopt

from scipy.optimize import  minimize


def _pack(coefs_, intercepts_):
    """Pack the parameters into a single vector."""
    return np.hstack([l.ravel() for l in coefs_ + intercepts_])


def inplace_softmax_derivative(Z, delta):
    sm_ = np.zeros(Z.shape + (Z.shape[1],))
    #eps = np.finfo(Z.dtype).eps
    for i in range(Z.shape[0]):
        s = Z[i].reshape(-1,1)
        sm_[i] = np.diagflat(s) - np.dot(s, s.T)  
        #sm_[i] = np.clip(sm_[i], -1 + eps, 1 - eps)      
        
        delta[i] = np.dot(sm_[i],delta[i].reshape(-1,1)).flatten()
        
    #print("sm min: ", sm_.min(), "sm max: ", sm_.max())    

DERIVATIVES = {
    "identity": inplace_identity_derivative,
    "tanh": inplace_tanh_derivative,
    "logistic": inplace_logistic_derivative,
    "relu": inplace_relu_derivative,
    "softmax": inplace_softmax_derivative
}

_STOCHASTIC_SOLVERS = ["sgd", "adam"]

def obinary_log_loss(y_true, y_prob):

    return (
        -(xlogy(y_true, y_prob).sum() + xlogy(1 - y_true, 1 - y_prob).sum())
        / y_prob.shape[0]
    )

def binary_log_loss(y_true, y_prob):
    """Compute binary logistic loss for classification.

    This is identical to log_loss in binary classification case,
    but is kept for its use in multilabel case.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_prob : array-like of float, shape = (n_samples, 1)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    eps = np.finfo(y_prob.dtype).eps
    logistic_sigmoid(y_prob, out=y_prob)
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    return (
        -(xlogy(y_true, y_prob).sum() + xlogy(1 - y_true, 1 - y_prob).sum())
        / y_prob.shape[0]
    )

LOSS_FUNCTIONS = {
    "binary_log_loss": binary_log_loss,
    "original_binary_log_loss": obinary_log_loss,    
}

class BiasedRecurrentClassifier(MLPClassifier):
    def _prune(self, mask = []):
        mask = set(mask)
        self.n_layers_ -= len(mask)
        self.recurrent_hidden -= len(mask)
        self.coefs_ = [c for i,c in enumerate(self.coefs_) if i not in mask]
        self.intercepts_ = [c for i,c in enumerate(self.intercepts_) if i not in mask]
        self.activation = [c for i,c in enumerate(self.activation) if i not in mask]
        self.layer_units = [c for i,c in enumerate(self.layer_units) if i not in mask]
        
    def _initialize(self, y, layer_units, dtype):
        print("RNN Init has been called...")
        # set all attributes, allocate weights etc. for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.t_ = 0
        #self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i], layer_units[i + 1], dtype
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        if self.solver in _STOCHASTIC_SOLVERS:
            self.loss_curve_ = []
            self._no_improvement_count = 0
            if self.early_stopping:
                self.validation_scores_ = []
                self.best_validation_score_ = -np.inf
                self.best_loss_ = None
            else:
                self.best_loss_ = np.inf
                self.validation_scores_ = None
                self.best_validation_score_ = None        
    

    def dual_fit2(self,X,y,I, bias = None, par_lr = 1.0, recurrent_hidden = 3):
        self.max_iter = 300
        #self.n_iter_no_change = 1000
        self._no_improvement_count = 0
        
        self.mixed_mode = False
        self.recurrent_hidden = recurrent_hidden
        self.bias = bias
        self.par_lr = par_lr        
        print ("Par lr: ", self.par_lr)        
        if len(y.shape) < 3:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[2]            
        
        print("Fit I->W->Y: ")
        self.best_loss_ = np.inf
        self.loss_curve_ = []
        self._fit(X, y, I, incremental=False, fit_mask = None) #list(range(recurrent_hidden - 1, self.n_layers_ - 1)))

        self.bias = None
        mask1 = list(range(recurrent_hidden - 1))
        mixed_copy = copy.deepcopy(self)
        mixed_copy._prune(mask = mask1)
        mixed_copy.mixed_mode = True
        self.warm_start = True
        deltas = []
        
        for _ in range(len(self.layer_units)):
            tmp = []
            for _ in range(X.shape[1]):
                tmp.append([])
            deltas.append(tmp)        
    
        hidden_grad = self._get_delta(X, y, deltas, bias, predict_mask = None)[self.recurrent_hidden - 1]

        return mixed_copy, np.swapaxes(np.asarray(hidden_grad),0,1) 
    
    def draw_plots(self,X_,j,par_lr, mask1, dir_):
        for i in range(X_.shape[2]):
            #i = 40
            res = []
            res2 = []
            for x in np.arange(-10,10.05,0.05):
                if i == X_.shape[2] - 2:
                    x_ = np.concatenate([X_[:1,:,:-2],np.zeros((1,X_.shape[1],2))],axis=2)
                    x_[0,j,i] = x
                    x_[0,j,i + 1] = 1. - x
                if i == X_.shape[2] - 1:
                    x_ = np.concatenate([X_[:1,:,:-2],np.zeros((1,X_.shape[1],2))],axis=2)
                    x_[0,j,i] = 1. - x
                    x_[0,j,i - 1] = x
                    
                if i < X_.shape[2] - 2:
                    x_ = X_[:1,:,:]
                    x_[0,j,i] = x        
                    
                activations = [x_]
                for _ in range(len(self.layer_units) - 1):
                    activations.append([])     
                self.mixed_mode = True             
                activations_ = self._forward_pass(activations, bias = None, par_lr = par_lr, predict_mask = mask1)
                self.mixed_mode = False
                 
                res.append(activations_[self.recurrent_hidden - 1][0,j,0])
                res2.append(activations_[self.recurrent_hidden - 1][0,j,1])
                     
            _, ax = plt.subplots()
             
            #print(res)
            ax.plot(list(np.arange(-10,10.05,0.05)), res)
            ax.plot(list(np.arange(-10,10.05,0.05)), res2)     
            plt.savefig(dir_ + "/network" + str(i)+ ".png")           
    
    
    def mockup_fit(self,X,y,X_,I, bias = None, par_lr = 1.0, recurrent_hidden = 3, imp_feature = None):
        self.validation_fraction = 0.1
        mask1 = list(range(recurrent_hidden - 1))
        self._no_improvement_count = 0
        self.best_loss_ = np.inf
        self.loss_curve_ = []        
        
        self.recurrent_hidden = recurrent_hidden
        self.bias = bias
        self.par_lr = par_lr
        print ("Par lr: ", self.par_lr)
        if len(y.shape) < 3:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[2]          

        self.mixed_mode = True
        
        ### copied from _fit
        # Make sure self.hidden_layer_sizes is a list
# 
#         if np.any(np.array(hidden_layer_sizes) <= 0):
#             raise ValueError(
#                 "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
#             )
#         first_pass = not hasattr(self, "coefs_") or (
#             not self.warm_start 
#         )
# 
#         n_features = X.shape[2]
# 
#         if self.n_outputs_ is None:
#             if len(y.shape) < 3:
#                 self.n_outputs_ = 1
#             else:
#                 self.n_outputs_ = y.shape[2]    
# 
#         if not self.mixed_mode:
#             self.layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]
#         else:    
#             self.layer_units = [n_features + hidden_layer_sizes[len(hidden_layer_sizes) - 1]] + hidden_layer_sizes + [self.n_outputs_]
# 
#         # check random state
#         self._random_state = check_random_state(self.random_state)
# 
#         if first_pass:
#             # First time training the model
#             self._initialize(y, self.layer_units, X.dtype)
# 
#         # Initialize lists
#         activations = [X_]
#         for _ in range(len(self.layer_units) - 1):
#             activations.append([])
#             
#         deltas = []
#         for _ in range(len(activations) - 1):
#             tmp = []
#             for _  in range(X.shape[1]):
#                 tmp.append([])      
#             deltas.append(tmp)
        
        #####
#         N = self.coefs_[0].shape[0] * self.coefs_[0].shape[1]
#         N += self.coefs_[1].shape[0] * self.coefs_[1].shape[1]
#         N += self.intercepts_[0].shape[0] + self.intercepts_[1].shape[0]
#         
#         counter = 0
#         
#         print("N: ", N)
# 
#         
#         def opt_func1(coefs_, grad = None):
#             nonlocal counter
#             nonlocal activations
#             nonlocal bias
#             nonlocal par_lr
#             nonlocal mask1
#             
#             #idx = np.random.randint(0,X_.shape[0],20)
#             
#             # Initialize lists
#             activations = [X_]
#             for _ in range(len(self.layer_units) - 1):
#                 activations.append([])            
# 
#             n = self.coefs_[0].shape[0] * self.coefs_[0].shape[1] 
#             self.coefs_[0] = coefs_[:n].reshape(self.coefs_[0].shape)
#             self.coefs_[1] = coefs_[n:n2].reshape(self.coefs_[1].shape)
#             self.intercepts_[0] = coefs_[n2:n2 + self.intercepts_[0].shape[0]]
#             self.intercepts_[1] = coefs_[n2 + self.intercepts_[0].shape[0]:]               
#             
#             activations_ = self._forward_pass(activations, bias = bias, par_lr = par_lr, predict_mask = mask1)
#             out = activations_[recurrent_hidden - 1]
#             self.loss = 'original_binary_log_loss'
#             
#             l =  LOSS_FUNCTIONS[self.loss](I.flatten(), out.flatten())
#             values = 0
#             for s in self.coefs_:
#                 s = s.ravel()
#                 values += np.dot(s, s)            
#             
#             l2 =(0.5 * float(1./100)) * values / activations[0].shape[0]
#             
#             
#             if counter % 1000 == 0:
#                 encoded_classes = np.asarray(out.flatten() >= 0, dtype=int)
#                 
#                 print("Acc: ", accuracy_score(encoded_classes, I.flatten()))                   
#                 
#                 print(counter,l,l2,l + l2)
#             
#             counter += 1
#             return l + l2
#         
#         opt = nlopt.opt(nlopt.GN_ESCH, N)
#         opt.set_min_objective(opt_func1)
#         opt.set_xtol_rel(0.01)
#          
#         opt.set_lower_bounds(-100)
#         opt.set_upper_bounds(100)     
#         opt.set_maxeval(50000)   
#         
#         init_x = np.hstack([self.coefs_[0].flatten(),self.coefs_[1].flatten(),self.intercepts_[0].flatten(),self.intercepts_[1].flatten()]) 
#         n = self.coefs_[0].shape[0] * self.coefs_[0].shape[1] 
#         n2 = n + self.coefs_[1].shape[0] * self.coefs_[1].shape[1]
#         opt_coefs = opt.optimize(init_x)
#         
#         counter = 0 
#         opt = nlopt.opt(nlopt.LN_BOBYQA, N)
#         opt.set_min_objective(opt_func1)
#         opt.set_xtol_rel(0.01)
#          
#         opt.set_lower_bounds(-100)
#         opt.set_upper_bounds(100)       
#         opt_coefs = opt.optimize(opt_coefs)
        
        
        #opt_coefs  = minimize(opt_func1,init_x, method = 'COBYLA')
        
#         self.coefs_[0] = opt_coefs[:n].reshape(self.coefs_[0].shape)
#         self.coefs_[1] = opt_coefs[n:n2].reshape(self.coefs_[1].shape)
#         self.intercepts_[0] = opt_coefs[n2:n2 + self.intercepts_[0].shape[0]]
#         self.intercepts_[1] = opt_coefs[n2 + self.intercepts_[0].shape[0]:]        
        
        self.max_iter = 10000
        self.learning_rate_init=0.00001
        self.alpha=1.
        print ("Fit X->I:")
        
        self._fit(X_, I, I, incremental=False, fit_mask = mask1, predict_mask = mask1)  
        self.warm_start = True
        
#         if imp_feature:
#             i = 40
#             print(i,X_[:,:,i].max(),X_[:,:,i].min())
#             res = []
#             res2 = []
#             for x in np.arange(0,1.05,0.05):
#                 x_ = np.concatenate([X_[:1,:,:-2],np.zeros((1,X_.shape[1],2))],axis=2)
#                 x_[0,0,i] = x
#                 x_[0,0,i + 1] = 1. - x
#                 activations = [x_]
#                 for _ in range(len(self.layer_units) - 1):
#                     activations.append([])                  
#                 activations_ = self._forward_pass(activations, bias = None, par_lr = par_lr, predict_mask = mask1)
#                 
#                 if i ==40 and np.abs(x  - 1.) < 0.01:
#                     print("Syntetic input")
#                     for i_,a in enumerate(activations_):
#                         print(i_,a)                        
#                     
#                 res.append(activations_[recurrent_hidden - 1][0,0,0])
#                 res2.append(activations_[recurrent_hidden - 1][0,0,1])
#                     
#             _, ax = plt.subplots()
#             
#             #print(res)
#             ax.plot(list(np.arange(0,1.05,0.05)), res)
#             ax.plot(list(np.arange(0,1.05,0.05)), res2)     
#             plt.savefig("network" + str(i)+ ".png")           
#         
        activations = [X_]
        for _ in range(len(self.layer_units) - 1):
            activations.append([])            
  
        activations_ = self._forward_pass(activations, bias = bias, par_lr = par_lr, predict_mask = mask1)  
         
        out = activations_[recurrent_hidden - 1]
        
        encoded_classes = np.asarray(out.flatten() >= 0.5, dtype=int)
                 
        print("Acc I: ", accuracy_score(encoded_classes, I.flatten()))         
        
#         
#         
#         print("Out: ", out.min(), out.max())
#         diff = np.abs(I.flatten() -  out.flatten())
#         print("diff train: ", diff.min(), diff.max(), diff.mean())
 
        self.warm_start = True   
        
        self.mixed_mode = False
        n_features = X.shape[2]
        

        self.layer_units = [n_features] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        
        
        activations = [X] + [None] * (len(self.layer_units) - 1)     
        
        counter = 0   
        
        print("Fit I->W->Y: ")

        def opt_func2(coefs_, grad):
            nonlocal counter
            nonlocal activations
            nonlocal bias
            nonlocal par_lr
            nonlocal mask1            
            
            n = self.coefs_[2].shape[0] * self.coefs_[2].shape[1] 
            n2 = n + self.coefs_[3].shape[0] * self.coefs_[3].shape[1]  
            self.coefs_[2] = coefs_[:n].reshape(self.coefs_[2].shape)
            self.coefs_[3] = coefs_[n:n2].reshape(self.coefs_[3].shape)
            self.intercepts_[2] = coefs_[n2:n2 + self.intercepts_[2].shape[0]]
            self.intercepts_[3] = coefs_[n2 + self.intercepts_[2].shape[0]:]   
            self.loss = 'binary_log_loss'           
            
            activations = self._forward_pass(activations, bias = bias, par_lr = par_lr)
            out = activations[len(activations) - 1]
            
           
            l = LOSS_FUNCTIONS[self.loss](y.flatten(), out.flatten())
            values = 0
            for s in self.coefs_:
                s = s.ravel()
                values += np.dot(s, s)            
             
            l2 =(0.5 * float(1./10000)) * values / activations[0].shape[0]            
            
            if counter % 1000 == 0:
                encoded_classes = np.asarray(out.flatten() >= 0, dtype=int)
                diff = np.abs(y.flatten() -  out.flatten())
                print("diff: ", diff.min(), diff.max(), diff.mean())
                print("Acc: ", accuracy_score(encoded_classes, y.flatten()))                   
                print(counter,l,l2) 
            counter += 1                               
            return l + l2
        
        N = self.coefs_[2].shape[0] * self.coefs_[2].shape[1]
        N += self.coefs_[3].shape[0] * self.coefs_[3].shape[1]        
        N += self.intercepts_[2].shape[0] + self.intercepts_[3].shape[0]
         
        counter = 0
         
        print("N: ", N)        
        opt = nlopt.opt(nlopt.LN_BOBYQA, N)
        opt.set_min_objective(opt_func2)
        opt.set_xtol_rel(0.01)  
        opt.set_maxeval(3000) 
#         opt.set_lower_bounds(-10000.)
#         opt.set_upper_bounds(10000.)           
        
        init_x = np.hstack([self.coefs_[2].flatten(),self.coefs_[3].flatten(),self.intercepts_[2].flatten(),self.intercepts_[3].flatten()])       
        
        opt_coefs = opt.optimize(init_x)
        n = self.coefs_[2].shape[0] * self.coefs_[2].shape[1]     
        n2 = n + self.coefs_[3].shape[0] * self.coefs_[3].shape[1]    
        self.coefs_[2] = opt_coefs[:n].reshape(self.coefs_[2].shape)
        self.coefs_[3] = opt_coefs[n:n2].reshape(self.coefs_[3].shape)   
        self.intercepts_[2] = opt_coefs[n2:n2 + self.intercepts_[2].shape[0]]
        self.intercepts_[3] = opt_coefs[n2 + self.intercepts_[2].shape[0]:]          
        
        self.bias = None
        
        mixed_copy = copy.deepcopy(self)
        mixed_copy._prune(mask = mask1)
        mixed_copy.mixed_mode = True
        
        deltas = []
        
        for _ in range(len(self.layer_units)):
            tmp = []
            for _ in range(X.shape[1]):
                tmp.append([])
            deltas.append(tmp)        
    
        hidden_grad = self._get_delta(X, y, deltas, bias, predict_mask = None)[self.recurrent_hidden - 1]

        return mixed_copy, np.swapaxes(np.asarray(hidden_grad),0,1) 
        
    
    def dual_fit(self,X,y,X_,I, bias = None, par_lr = 1.0, recurrent_hidden = 3):
        
        self._no_improvement_count = 0
        self.best_loss_ = np.inf
        self.loss_curve_ = []        
        
        self.mixed_mode = False
        self.recurrent_hidden = recurrent_hidden
        self.bias = bias
        self.par_lr = par_lr
        print ("Par lr: ", self.par_lr)
        if len(y.shape) < 3:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[2]         
        
        mask1 = list(range(recurrent_hidden - 1))
        #self.n_iter_no_change = 50
        self.max_iter = 5
        self.learning_rate_init=0.001
        self.alpha=1./100.
        print ("Fit X->I:")
        self.mixed_mode = True
        self._fit(X_, I, I, incremental=False, fit_mask = mask1, predict_mask = mask1)
        self.mixed_mode = False
        #ws_tmp = self.warm_start
        self.warm_start = True
        self.max_iter = 5
        #self.n_iter_no_change = 1000
        self._no_improvement_count = 0
        print("Fit I->W->Y: ")
        self.best_loss_ = np.inf
        self.loss_curve_ = []
        #X = X[:,:,20:]
        self.learning_rate_init=0.00001
        self.alpha=1./100.
        self._fit(X, y, I,incremental=False, fit_mask = list(range(recurrent_hidden - 1, self.n_layers_ - 1)))
        #self.warm_start = ws_tmp
        self.bias = None
        
        mixed_copy = copy.deepcopy(self)
        mixed_copy._prune(mask = mask1)
        mixed_copy.mixed_mode = True
        
        deltas = []
        
        for _ in range(len(self.layer_units)):
            tmp = []
            for _ in range(X.shape[1]):
                tmp.append([])
            deltas.append(tmp)        
    
        hidden_grad = self._get_delta(X, y, deltas, bias, predict_mask = None)[self.recurrent_hidden - 1]

        return mixed_copy, np.swapaxes(np.asarray(hidden_grad),0,1) 

    def _forward_pass(self, activations, bias = None, par_lr = 1.0, predict_mask = None):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        if predict_mask is not None:
            layer_range_all = predict_mask + [predict_mask[len(predict_mask) - 1] + 1]
        else:    
            layer_range_all = list(range(self.n_layers_))
                
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = np.zeros((activations[0].shape[0],activations[0].shape[1],self.layer_units[i + 1]))    
        
        for t in range(activations[0].shape[1]):
            for n,i in enumerate(layer_range_all[:-1]):
                next_i = layer_range_all[n + 1]
                hidden_activation = ACTIVATIONS[self.activation[i]]
                if i == 0 and not self.mixed_mode:
                    if t > 0:
                        #activations[i + 1][:,t] = safe_sparse_dot(np.hstack([activations[self.n_layers_ -  2][:,t - 1], activations[i][:,t]]), self.coefs_[i])
                        activations[next_i][:,t] = safe_sparse_dot(np.hstack([activations[self.recurrent_hidden][:,t - 1], activations[i][:,t]]), self.coefs_[i])
                    else:
                        #init_add = np.zeros((activations[i][:,t].shape[0],activations[self.n_layers_ - 2][:,t].shape[1])) 
                        init_add = np.zeros((activations[i][:,t].shape[0],activations[i + self.recurrent_hidden][:,t].shape[1]))
                        activations[next_i][:,t] = safe_sparse_dot(np.hstack([init_add, activations[i][:,t]]), self.coefs_[i])                            
                else:
                    activations[next_i][:,t] = safe_sparse_dot(activations[i][:,t], self.coefs_[i])    
                    
                activations[next_i][:,t] += self.intercepts_[i]
                
                # For the hidden layers
                if (next_i) != (self.n_layers_ - 1):
                    hidden_activation(activations[next_i][:,t])                  
                
                if bias is not None and next_i == self.recurrent_hidden:
                    activations[next_i][:,t] = par_lr * (activations[next_i][:,t]) + bias[:,t]#.reshape(-1,1) 
                        
        return activations    
    
    def merge(self, model):
        self.activation = model.activation + self.activation
        self.n_layers += model.n_layers
        self.coefs_ = model.coefs_ + self.coefs_
        self.intercepts_ = model.intercepts_ + self.intercepts_
    
    
    def fit(self, X, y, bias = None, par_lr = 1.0, recurrent_hidden = 0):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        self.bias = bias
        self.par_lr = par_lr

        self.recurrent_hidden = recurrent_hidden

        self.n_outputs_ = None
        self.mixed_mode = False
        res = self._fit(X, y, incremental=False)
        self.bias = None
        return res   
    
    def _score(self, X, y, bias):
        """Private score method without input validation"""
        # Input validation would remove feature names, so we disable it
        
        activations = [X]
        for _ in range(len(self.layer_units) - 1):
            activations.append([])     
            
        mask1 = list(range(self.recurrent_hidden - 1))
        
        activations_ = self._forward_pass(activations, bias = bias, par_lr = self.par_lr, predict_mask = mask1)
        out = activations_[self.recurrent_hidden - 1]               
        
        return LOSS_FUNCTIONS['original_binary_log_loss'](y.flatten(), out.flatten())      
    
    def _predict(self, X, check_input=True, bias = None):
        """Private predict method with optional input validation"""
        if check_input:
            X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)

        # Initialize first layer
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        _, n_features = X.shape

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)       
        
        activations = self._forward_pass(activations, bias)
        
        y_pred = activations[self.n_layers_ - 1]

        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()

        return self._label_binarizer.inverse_transform(y_pred), activations
    
    def predict_proba(self, X, check_input=True, get_non_activated = False, bias=None,par_lr = 1.0):
        """Private predict method with optional input validation"""
        # Initialize first layer
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        #n_features = X.shape[2]

        layer_units = self.layer_units

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)   
        
        if  get_non_activated:
            non_activations = activations.copy()
        else:
            non_activations = None       
        activations = self._forward_pass(activations, bias=bias, par_lr = par_lr)
        
        y_pred = activations[self.n_layers_ - 1]

        if self.n_outputs_ == 1:
            y_pred = y_pred.reshape((y_pred.shape[0],y_pred.shape[1]))
        if non_activations:
            return y_pred, activations[self.recurrent_hidden], non_activations[self.recurrent_hidden]
        else:     
            return y_pred, activations  
        
    def _loss_grad_lbfgs(
        self, packed_coef_inter, X, y, I, activations, deltas, coef_grads, intercept_grads, fit_mask = None, predict_mask = None
    ):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to the different parameters given in the initialization.

        Returned gradients are packed in a single vector so it can be used
        in lbfgs

        Parameters
        ----------
        packed_coef_inter : ndarray
            A vector comprising the flattened coefficients and intercepts.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        grad : array-like, shape (number of nodes of all layers,)
        """
        self._unpack(packed_coef_inter)
        loss, coef_grads, intercept_grads,_,_ = self._backprop(
            X, y, I, activations, deltas, coef_grads, intercept_grads,self.bias,fit_mask, predict_mask
        )
        grad = _pack(coef_grads, intercept_grads)
        return loss, grad        
        
    def _fit_lbfgs(
        self, X, y, I, activations, deltas, coef_grads, intercept_grads, layer_units, fit_mask = None, predict_mask = None
    ):
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        # Run LBFGS
        packed_coef_inter = _pack(self.coefs_, self.intercepts_)

        if self.verbose is True or self.verbose >= 1:
            iprint = 1
        else:
            iprint = -1

        opt_res = scipy.optimize.minimize(
            self._loss_grad_lbfgs,
            packed_coef_inter,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": self.max_fun,
                "maxiter": self.max_iter,
                "iprint": iprint,
                "gtol": self.tol,
            },
            args=(X, y, I, activations, deltas, coef_grads, intercept_grads, fit_mask, predict_mask),
        )
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
        self.loss_ = opt_res.fun
        self._unpack(opt_res.x)
        
    def _fit(self, X, y, I, incremental=False, fit_mask = None, predict_mask = None):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )


        n_features = X.shape[2]

        if self.n_outputs_ is None:
            if len(y.shape) < 3:
                self.n_outputs_ = 1
            else:
                self.n_outputs_ = y.shape[2]    

        if self.mixed_mode:
            self.layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]
        else:    
            self.layer_units = [n_features + hidden_layer_sizes[len(hidden_layer_sizes) - 1]] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, self.layer_units, X.dtype)

        # Initialize lists
        activations = [X]
        for _ in range(len(self.layer_units) - 1):
            activations.append([])
            
        deltas = []
        for _ in range(len(activations) - 1):
            tmp = []
            for _  in range(X.shape[1]):
                tmp.append([])      
            deltas.append(tmp)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(self.layer_units[:-1], self.layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in self.layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(
                X,
                y,
                I,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                self.layer_units,
                incremental,
                fit_mask, predict_mask
            )

        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X, y,I, activations, deltas, coef_grads, intercept_grads, self.layer_units, fit_mask, predict_mask
            )

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self        

    def _fit_stochastic(
        self,
        X,
        y,
        I,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
        fit_mask = None,
        predict_mask = None
    ):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        # early_stopping in partial_fit doesn't make sense
        if self.early_stopping and incremental:
            raise ValueError("partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping
        
        bias = self.bias
        if True:#early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            if self.bias is not None:
                X, X_val, y, y_val, I, I_val, bias,bias_val = train_test_split(
                    X,
                    y,
                    I,
                    self.bias,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=None,
                )
            else:
                X, X_val, y, y_val, I, I_val = train_test_split(
                    X,
                    y,
                    I,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=None,
                )                    
        else:
            X_val = None
            y_val = None
            bias_val = None
            I_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    # Only shuffle the sample indices instead of X and y to
                    # reduce the memory footprint. These indices will be used
                    # to slice the X and y.
                    sample_idx = shuffle(sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                al1 = 0.0
                al2 = 0.0
                bc = 0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        X_batch = _safe_indexing(X, sample_idx[batch_slice])
                        y_batch = y[sample_idx[batch_slice]]
                        bias_batch = bias[sample_idx[batch_slice]]
                        I_batch = I[sample_idx[batch_slice]]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]
                        bias_batch = bias[batch_slice]
                        I_batch = I[batch_slice]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads, bl1, bl2 = self._backprop(
                        X_batch,
                        y_batch,
                        I_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                        bias_batch,
                        fit_mask,
                        predict_mask
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )
                    
                    al1 += bl1 * (
                        batch_slice.stop - batch_slice.start
                    )
                    
                    al2 += bl2 * (
                        batch_slice.stop - batch_slice.start
                    )
                     
                    bc += 1
                    #if it == 0 and bc < 20:
                    #    print("Acc loss: ", accumulated_loss / (batch_size * bc))
                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]
                
                l1 = al1 / X.shape[0]
                l2 = al2 / X.shape[0]
                
                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_),l1,l2,"val loss: ",self._score(X_val, I_val, bias_val))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_),l1,l2,"val loss: ",self._score(X_val, I_val, bias_val))
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_),l1,l2,"val loss: ",self._score(X_val, I_val, bias_val))                    
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter,
                        ConvergenceWarning,
                    )
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts
            self.validation_scores_ = self.validation_scores_ 
            
    def _compute_loss_grad(
        self,t, layer, n_samples, activations, deltas, coef_grads, intercept_grads
    ):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.

        This function does backpropagation for the specified one layer.
        """
        if layer == 0 and not self.mixed_mode:
            if t > 0:
                #sm = safe_sparse_dot(np.hstack([activations[self.n_layers_ -  2][:,t - 1],activations[layer][:,t]]).T, deltas[layer][t])
                sm = safe_sparse_dot(np.hstack([activations[layer + self.recurrent_hidden][:,t - 1],activations[layer][:,t]]).T, deltas[layer][t])
            else:
                init_add = np.zeros((activations[layer][:,t].shape[0],activations[layer + self.recurrent_hidden][:,t].shape[1]))
                sm = safe_sparse_dot(np.hstack([init_add, activations[layer][:,t]]).T, deltas[layer][t])    
        else:    
            sm = safe_sparse_dot(activations[layer][:,t].T, deltas[layer][t])
        sm += self.alpha * self.coefs_[layer]
        sm /= n_samples
        coef_grads[layer] += sm

        intercept_grads[layer] += np.mean(deltas[layer][t], 0)    
        
    def _get_delta(self, X, y, deltas, bias, predict_mask = None):
        # Forward propagate
        activations = [X] + [None] * (len(self.layer_units) - 1)  
        activations = self._forward_pass(activations, bias, par_lr = self.par_lr)#, mask = predict_mask)

        if predict_mask is not None:
            layer_range = sorted(list(predict_mask),reverse=True)
        else:    
            layer_range = list(range(self.n_layers_ - 2, -1, -1))

        # Backward propagate
        last = layer_range[0]

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        
        for t in range(X.shape[1] - 1, -1, -1):
            if last == self.n_layers_ - 2:
                eps = np.finfo(activations[last][:,t].dtype).eps
                y_prob = logistic_sigmoid(activations[last + 1][:,t])
                y_prob = np.clip(y_prob, eps, 1 - eps)
            else:
                y_prob = activations[last + 1][:,t]
                    
            deltas[last][t] = (y_prob - y[:,t].reshape(y_prob.shape))#.reshape(-1,1)
    
    
            # Iterate over the hidden layers
            for n,i in enumerate(layer_range[:-1]):
                prev_i = layer_range[n + 1]
                inplace_derivative = DERIVATIVES[self.activation[i - 1]]
                #if i == self.n_layers_ -  2 and t < X.shape[1] - 1:
                if i == self.recurrent_hidden and t < X.shape[1] - 1:
                    deltas[prev_i][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                    deltas[prev_i][t] += safe_sparse_dot(deltas[0][t + 1],self.coefs_[0][:deltas[i][t].shape[1],:].T)
                else:    
                    deltas[prev_i][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                inplace_derivative(activations[i][:,t], deltas[prev_i][t])        
                
        return deltas  
          
    def _backprop(self, X, y, I, activations, deltas, coef_grads, intercept_grads, bias, fit_mask = None, predict_mask = None):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.

        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function

        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.

        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]
        for c in coef_grads:
            c.fill(0.)
        for c in intercept_grads:     
            c.fill(0.)

        # Forward propagate
        activations = self._forward_pass(activations, bias, par_lr = self.par_lr, predict_mask = predict_mask)

        if predict_mask is not None:
            layer_range = sorted(list(predict_mask),reverse=True)
        else:    
            layer_range = list(range(self.n_layers_ - 2, -1, -1))

        # Backward propagate
        last = layer_range[0]

        # Get loss
        loss_func_name = self.loss

        if last != self.n_layers_ - 2:
            loss_func_name = "original_binary_log_loss"
        else:
            loss_func_name = "binary_log_loss"    
        
        loss_ = LOSS_FUNCTIONS[loss_func_name](y.flatten(), activations[last + 1].flatten())
        #loss2 = LOSS_FUNCTIONS["original_binary_log_loss"](I.flatten(), activations[self.recurrent_hidden - 1].flatten())
        
        #loss = loss_ + loss2

        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss2 = (0.5 * self.alpha) * values / n_samples
        
        loss = loss_ + loss2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        
        for t in range(X.shape[1] - 1, -1, -1):
            if last == self.n_layers_ - 2:
                eps = np.finfo(activations[last][:,t].dtype).eps
                y_prob = logistic_sigmoid(activations[last + 1][:,t])
                y_prob = np.clip(y_prob, eps, 1 - eps)
            else:
                y_prob = activations[last + 1][:,t]
                    
            deltas[last][t] = (y_prob - y[:,t].reshape(y_prob.shape))#.reshape(-1,1)
            #deltas[last][t] = activations[-1][:,t] - y[:,t].reshape(-1,1)
    
            # Compute gradient for the last layer
            self._compute_loss_grad(
                t, last, n_samples, activations, deltas, coef_grads, intercept_grads
            )
    
            # Iterate over the hidden layers
            for n,i in enumerate(layer_range[:-1]):
                prev_i = layer_range[n + 1]
                inplace_derivative = DERIVATIVES[self.activation[i - 1]]
                #if i == self.n_layers_ -  2 and t < X.shape[1] - 1:
                   
                if i == self.recurrent_hidden and t < X.shape[1] - 1:
                    deltas[prev_i][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                    deltas[prev_i][t] += safe_sparse_dot(deltas[0][t + 1],self.coefs_[0][:deltas[i][t].shape[1],:].T)
                else:    
                    deltas[prev_i][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                    
                #if i == self.recurrent_hidden - 1:
                #    deltas[prev_i][t] += 0.1 * (activations[i][:,t] - I[:,t].reshape(activations[i][:,t].shape))    
                    
                inplace_derivative(activations[i][:,t], deltas[prev_i][t])        
                
                #if i >= 3:
                self._compute_loss_grad(
                    t, prev_i, n_samples, activations, deltas, coef_grads, intercept_grads
                    )
        if fit_mask is not None:
            rem = set(range(self.n_layers_ - 1)).difference(fit_mask)
            for r in rem:
                coef_grads[r] = np.zeros(coef_grads[r].shape)
                intercept_grads[r] = np.zeros(intercept_grads[r].shape)     
        return loss, coef_grads, intercept_grads, loss_, loss2                      