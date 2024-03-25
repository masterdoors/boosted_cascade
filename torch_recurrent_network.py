import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import copy

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
        assert len(layer_units) == len(activations) + 1
        super(TorchRNN, self).__init__()
        self.activations = activations
        self.recurrent_hidden = recurrent_hidden
        self.layers = nn.ModuleList([])  
        self.layer_units = layer_units
        for frm, too  in zip(layer_units[:-1],layer_units[1:]):
            self.layers.append(nn.Linear(frm, too))

  
    def forward(self, x, hidden_state, predict_mask, bias,par_lr):
        hidden = []
        if hidden_state is not None:
            res = torch.cat((hidden_state,x), 1)
        else:
            res = x
            
        if predict_mask is None:
            predict_mask = list(range(len(self.layers)))    
                
        for i in predict_mask:
            if self.activations[i] != "identity":
                res = TorchRNN.ACTIVATIONS[self.activations[i]](self.layers[i](res)) 
            else:
                res = self.layers[i](res)   
                
            if i == self.recurrent_hidden - 1:
                res = par_lr * res + bias
            hidden.append(res)
        return res, hidden           
    
    def init_hidden(self, batch):
        return nn.init.kaiming_uniform_(torch.empty(batch, self.layer_units[self.recurrent_hidden]))
    
    
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
        torch.set_default_dtype(torch.double)
        self.activation = activation  
        self.verbose = verbose 
        self.tol = tol 
        self.max_fun = max_fun
        if batch_size == 'auto':
            batch_size = 10
        else:    
            self.batch_size = batch_size
            
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.model = None
        self.tree_approx_data_size = 30000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_mode = False
        
    def _prune(self, mask = []):
        mask = set(mask)
        self.n_layers -= len(mask)
        self.model.recurrent_hidden -= len(mask)  
        self.model.layers = nn.ModuleList([c for i,c in enumerate(self.model.layers) if i not in mask])
        self.model.activations = [c for i,c in enumerate(self.model.activations) if i not in mask]
        self.layer_units = [c for i,c in enumerate(self.layer_units) if i not in mask]   
          
    
    def get_coefs_(self,i):
        return self.model.layers[i].weight.detach().numpy().T   
    
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
            
        self.layer_units = [n_features + self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1]] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        self.n_layers = len(self.layer_units)  
        
        if self.model is None:
            self.model = TorchRNN(self.layer_units, self.activation, recurrent_hidden = recurrent_hidden)
            self.model.to(device=self.device)
            
        self.learning_rate_init = 0.001  
        self.alpha = 0.0000  
        self.max_iter = 100
        X_add, I_add = self.sampleXIdata(T,X_,self.tree_approx_data_size)    
        criterion = nn.BCELoss()
        self.mixed_mode = True
        self._fit(torch.from_numpy(np.vstack([X_,X_add])).to(device=self.device), torch.from_numpy(np.vstack([I,I_add])).to(device=self.device), criterion, fit_mask = mask1, predict_mask = mask1)  
        
        self.alpha = 0.0001 
        #self.layer_units = [n_features] + list(self.hidden_layer_sizes) + [self.n_outputs_]
        self.learning_rate_init = 0.00001
        self.max_iter = 100
        criterion = nn.BCEWithLogitsLoss()
        self.mixed_mode = False
        self._fit(torch.from_numpy(X).to(device=self.device), torch.from_numpy(y).to(device=self.device), criterion, fit_mask = list(range(recurrent_hidden - 1, self.n_layers - 1)),bias = bias, par_lr = par_lr)
        
        deltas = []
        for _ in range(len(self.layer_units)):
            tmp = []
            for _ in range(X.shape[1]):
                tmp.append([])
            deltas.append(tmp)          
        
        hidden_state = self.model.init_hidden(batch=X.shape[0])
                
        activations = [X]
        for t in range(X.shape[1]):
            out, hidden = self.model(torch.from_numpy(X[:,t]).to(device=self.device), hidden_state, None,torch.from_numpy(bias[:,t]).to(device=self.device),par_lr)
            hidden_state = hidden[self.model.recurrent_hidden - 1]
            for i in range(1,len(hidden) + 1):
                if len(activations) < i + 1:
                    activations.append([])    
                activations[i].append(hidden[i - 1].detach().to(torch.device('cpu')).numpy())

        for i in range(1,len(activations)):
            activations[i] = np.swapaxes(np.asarray(activations[i]),0,1)        

        last = len(activations) - 2

        for t in range(X.shape[1] - 1, -1, -1):
            eps = np.finfo(activations[last + 1][:,t].dtype).eps
            y_prob = logistic_sigmoid(activations[last + 1][:,t])
            y_prob = np.clip(y_prob, eps, 1 - eps)

                    
            deltas[last][t] = (y_prob - y[:,t].reshape(y_prob.shape))#.reshape(-1,1)
    
    
            # Iterate over the hidden layers
            for i in range(last,-1,-1):

                inplace_derivative = DERIVATIVES[self.activation[i - 1]]
                #if i == self.n_layers_ -  2 and t < X.shape[1] - 1:
                if i == recurrent_hidden and t < X.shape[1] - 1:
                    deltas[i - 1][t] = safe_sparse_dot(deltas[i][t], self.get_coefs_(i).T)
                    deltas[i - 1][t] += safe_sparse_dot(deltas[0][t + 1],self.get_coefs_(0)[:deltas[i][t].shape[1],:].T)
                else:    
                    deltas[i - 1][t] = safe_sparse_dot(deltas[i][t], self.get_coefs_(i).T)
                inplace_derivative(activations[i][:,t], deltas[i - 1][t])   
                
        mixed_copy = copy.deepcopy(self)
        mixed_copy._prune(mask = mask1)
        mixed_copy.mixed_mode = True                     
                
        return mixed_copy, deltas         
          

                
    def _fit(self,X,y,criterion,fit_mask = None, predict_mask = None, bias = None, par_lr = 1.):   
        for i,param in enumerate(self.model.parameters()):
            if i not in fit_mask:
                param.requires_grad = False
            else:        
                param.requires_grad = True
                       
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)

        for epoch in range(self.max_iter):
            #np.random.shuffle(X)
            eloss = 0.
            eacc = 0.
            for i in range(int(X.shape[0] / self.batch_size)):
                batch_idxs = np.random.randint(0,X.shape[0],self.batch_size)
                X_batch =  X[batch_idxs]
                y_batch = y[batch_idxs]
                
                if bias is not None:
                    bias_batch = torch.from_numpy(bias[batch_idxs]).to(self.device)
                else:
                    bias_batch = None
                         
                output = []
                if self.mixed_mode:
                    for t in range(X.shape[1]):
                        out, _ = self.model(X_batch[:,t], None, predict_mask,bias_batch,par_lr)
                        output.append(out)                    
                else:      
                    hidden_state = self.model.init_hidden(self.batch_size)
                    
                    for t in range(X.shape[1]):
                        out, hidden = self.model(X_batch[:,t], hidden_state, predict_mask,bias_batch[:,t],par_lr)
                        hidden_state = hidden[self.model.recurrent_hidden - 1]
                        output.append(out)
            
                loss = criterion(torch.hstack(output).flatten(), y_batch.flatten())
                eloss += loss.item()
                encoded_classes_ = np.asarray(torch.hstack(output).flatten().detach().to(torch.device('cpu')).numpy() >= 0.5, dtype=int)
                acc = accuracy_score(encoded_classes_,y_batch.flatten().detach().numpy())  
                eacc += acc              
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                
            if self.verbose:

                print(
                    f"Epoch [{epoch + 1}/{self.max_iter}], "
                    f"Loss: {eloss / int(X.shape[0] / self.batch_size):.4f}",
                    eacc / int(X.shape[0] / self.batch_size)
                )          
    
    def predict_proba(self, X, check_input=True, get_non_activated = False, bias=None,par_lr = 1.0):
        with torch.no_grad():
            hidden_state = self.model.init_hidden(batch=X.shape[0])
            output = []
            activations = [X]
            if bias is not None:
                bias = torch.from_numpy(bias).to(self.device)     
            for t in range(X.shape[1]):
                if self.mixed_mode:
                    out, hidden = self.model(torch.from_numpy(X[:,t]).to(self.device), None,None, bias[:,t], par_lr)
                else:    
                    out, hidden = self.model(torch.from_numpy(X[:,t]).to(self.device), hidden_state, None,bias[:,t], par_lr)
                    hidden_state = hidden[self.model.recurrent_hidden - 1]
                output.append(out.detach().to(torch.device('cpu')).numpy())
                for i in range(1,len(hidden) + 1):
                    if len(activations) < i + 1:
                        activations.append([])    
                    activations[i].append(hidden[i - 1].detach().to(torch.device('cpu')).numpy())

            for i in range(1,len(activations)):
                activations[i] = np.swapaxes(np.asarray(activations[i]),0,1)  
        return np.hstack(output).reshape(X.shape[0],X.shape[1]), activations        
                
                
