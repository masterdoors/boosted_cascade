import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        hidden = None
        if hidden_state:
            res = torch.cat((hidden_state,x), 1)
        else:
            res = x
                
        for i in predict_mask:
            res = TorchRNN.ACTIVATIONS[self.activations[i]](self.layers(res)) 
            if i == self.recurrent_hidden:
                hidden = res
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
        
        if self.model is None:
            self.model = TorchRNN(self.layer_units, self.activation)
            
        self.learning_rate_init = 0.001    
        X_add, I_add = self.sampleXIdata(T,X_,self.tree_approx_data_size)    
        self._fit(np.vstack([X_,X_add]), np.vstack([I,I_add]), incremental=False, fit_mask = mask1, predict_mask = mask1)  
        
        
        self.layer_units = [n_features] + self.hidden_layer_sizes + [self.n_outputs_]
        self.learning_rate_init = 0.0001
        self._fit(X, y, incremental=False, fit_mask = list(range(recurrent_hidden - 1, self.n_layers_ - 1)))
        #TODO get deltas
          

                
    def _fit(self,X,y,fit_mask = None, predict_mask = None):   
        for i,param in self.model.parameters():
            if i not in fit_mask:
                param.requires_grad = False
            else:        
                param.requires_grad = True
                       
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
        
        for epoch in range(self.max_iter):
            np.random.shuffle(X)
            for i in range(int(X.shape[0] / self.batch_size)):
                batch_idxs = np.random.randint(0,X.shape[0],self.batch_size)
                X_batch =  X[batch_idxs]
                y_batch = y[batch_idxs]
                
                hidden_state = self.model.init_hidden()
                
                output, hidden_state = self.model(X_batch, hidden_state, predict_mask)
                loss = criterion(output, y_batch)
        
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
      
    
    def _get_delta(self, X, y, deltas, bias, predict_mask = None):
        pass