import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from sklearn.neural_network._base import inplace_identity_derivative,inplace_tanh_derivative, inplace_logistic_derivative, inplace_relu_derivative
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import expit as logistic_sigmoid

def inplace_softmax_derivative(Z, delta):
    sm_ = np.zeros(Z.shape + (Z.shape[1],))
    #eps = np.finfo(Z.dtype).eps
    for i in range(Z.shape[0]):
        s = Z[i].reshape(-1,1)
        sm_[i] = np.diagflat(s) - np.dot(s, s.T)  
        #sm_[i] = np.clip(sm_[i], -1 + eps, 1 - eps)      
        
        delta[i] = np.dot(sm_[i],delta[i].reshape(-1,1)).flatten()


DERIVATIVES = {
    "identity": inplace_identity_derivative,
    "tanh": inplace_tanh_derivative,
    "logistic": inplace_logistic_derivative,
    "relu": inplace_relu_derivative,
    "softmax": inplace_softmax_derivative
}

class TorchRNN(nn.Module):
    ACTIVATIONS = {
        "identity": torch.nn.Identity,
        "tanh": torch.tanh,
        "logistic": torch.sigmoid,
        "relu": torch.relu,
        "softmax": torch.softmax,
    }
    
    
    def __init__(self, layer_units, activations, recurrent_hidden):
        assert len(layer_units) == len(activations)
        super(TorchRNN, self).__init__()
        self.activations = activations
        self.recurrent_hidden = recurrent_hidden
        self.layers = []  
        for frm, too  in layer_units:
            self.layers.append(nn.Linear(frm, too))

  
    def forward(self, x, hidden_state, predict_mask):
        hidden = []
        if hidden_state:
            res = torch.cat((hidden_state,x), 1)
        else:
            res = x
            
        if predict_mask is None:
            predict_mask = list(range(len(self.layers)))    
                
        for i in predict_mask:
            res = TorchRNN.ACTIVATIONS[self.activations[i]](self.layers(res)) 
            hidden.append(res)
        return res, hidden           
    
    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
    
    
    
class BiasedRecurrentClassifier:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation  
        self.verbose = verbose 
        self.tol = tol 
        self.max_fun = max_fun
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.model = None
        self.tree_approx_data_size = 30000
    
    def get_coefs_(self,i):
        return self.model.layers[i].weight   
    
    coefs_ = property(fget = get_coefs_)
        
    def sampleXIdata(self,T,X,size):
        rnd_part = np.random.random((size,X.shape[1],X.shape[2]))
        min_ = X.reshape(-1,X.shape[2]).min(axis=0)
        max_ = X.reshape(-1,X.shape[2]).max(axis=0)
        diff = (max_ - min_) + 0.0001
        X_ = rnd_part * diff + min_
        I =  T.getIndicators(T.forest, X_.reshape(-1,X.shape[2]), False, False)
        
        return X_,I.reshape(X_.shape[0],X_.shape[1],I.shape[1])        
    
    
    def mockup_fit(self,X,y,X_,I, T, bias = None, par_lr = 1.0, recurrent_hidden = 3, imp_feature = None):
        n_features = X.shape[2]
        mask1 = list(range(recurrent_hidden - 1))
        if len(y.shape) < 3:
            self.n_outputs_ = 1
        else:    
            self.n_outputs_ = y.shape[2]
            
        self.layer_units = [n_features + self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1]] + self.hidden_layer_sizes + [self.n_outputs_]
        self.n_layers = len(self.layer_units)  
        
        if self.model is None:
            self.model = TorchRNN(self.layer_units, self.activation)
            
        self.learning_rate_init = 0.001    
        X_add, I_add = self.sampleXIdata(T,X_,self.tree_approx_data_size)    
        self._fit(np.vstack([X_,X_add]), np.vstack([I,I_add]), incremental=False, fit_mask = mask1, predict_mask = mask1)  
        
        
        self.layer_units = [n_features] + self.hidden_layer_sizes + [self.n_outputs_]
        self.learning_rate_init = 0.0001
        self._fit(X, y, incremental=False, fit_mask = list(range(recurrent_hidden - 1, self.n_layers_ - 1)))
        
        deltas = []
        for _ in range(len(self.layer_units)):
            tmp = []
            for _ in range(X.shape[1]):
                tmp.append([])
            deltas.append(tmp)          
        
        
        hidden_state = self.model.init_hidden()
                
        activations = [X]
        for t in range(X.shape[1]):
            out, hidden = self.model(X[:,t], hidden_state)
            hidden_state = hidden[self.model.recurrent_hidden]
            activations.append(out)

        last = len(activations) - 2

        for t in range(X.shape[1] - 1, -1, -1):
            eps = np.finfo(activations[last][:,t].dtype).eps
            y_prob = logistic_sigmoid(activations[last + 1][:,t])
            y_prob = np.clip(y_prob, eps, 1 - eps)

                    
            deltas[last][t] = (y_prob - y[:,t].reshape(y_prob.shape))#.reshape(-1,1)
    
    
            # Iterate over the hidden layers
            for i in range(last - 1,-1,-1):

                inplace_derivative = DERIVATIVES[self.activation[i - 1]]
                #if i == self.n_layers_ -  2 and t < X.shape[1] - 1:
                if i == self.recurrent_hidden and t < X.shape[1] - 1:
                    deltas[i - 1][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                    deltas[i - 1][t] += safe_sparse_dot(deltas[0][t + 1],self.coefs_[0][:deltas[i][t].shape[1],:].T)
                else:    
                    deltas[i - 1][t] = safe_sparse_dot(deltas[i][t], self.coefs_[i].T)
                inplace_derivative(activations[i][:,t], deltas[i - 1][t])        
                
        return deltas         
          

                
    def _fit(self,X,y,fit_mask = None, predict_mask = None):   
        for i,param in self.model.parameters():
            if i not in fit_mask:
                param.requires_grad = False
            else:        
                param.requires_grad = True
                       
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init)

        for epoch in range(self.max_iter):
            np.random.shuffle(X)
            for i in range(int(X.shape[0] / self.batch_size)):
                batch_idxs = np.random.randint(0,X.shape[0],self.batch_size)
                X_batch =  X[batch_idxs]
                y_batch = y[batch_idxs]
                
                hidden_state = self.model.init_hidden()
                output = []
                for t in range(X.shape[1]):
                    out, hidden = self.model(X_batch[:,t], hidden_state, predict_mask)
                    hidden_state = hidden[self.model.recurrent_hidden]
                    output.append(out)
                loss = criterion(np.hstack(output), y_batch)
        
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                
            if self.verbose:
                print(
                    f"Epoch [{epoch + 1}/{self.max_iter}], "
                    f"Loss: {loss.item():.4f}"
                )          
    
    def predict_proba(self, X, check_input=True, get_non_activated = False, bias=None,par_lr = 1.0):
        self.model.eval()
        
        with torch.no_grad():
            for name, label in X:
                hidden_state = self.model.init_hidden()
                for char in name:
                    output, hidden_state = self.model(char, hidden_state)
