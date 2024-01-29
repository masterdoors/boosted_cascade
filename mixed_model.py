'''
Created on Jan 29, 2024

@author: keen
'''
import copy
import numpy as np
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices


class MixedModel:
    def __init__(self, forest_estimator, network_estimator, max_iter = 10,learning_rate = 1.):
        self.learning_rate = learning_rate
        self.forest = copy.deepcopy(forest_estimator)
        self.network_estimator = network_estimator
        self.network = copy.deepcopy(network_estimator)
        self.max_iter = max_iter
        
    def getIndicators(self, estimator, X, sampled = True, do_sample = True):
        Is = []
        n_samples = X.shape[0]
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples,
            estimator.max_samples,
        )  
        idx = estimator.apply(X)
        for i,clf in enumerate(estimator.estimators_):
            if do_sample:
                if sampled:
                    indices = _generate_sample_indices(
                        clf.random_state,
                        n_samples,
                        n_samples_bootstrap,
                    )        
                else:    
                    indices = _generate_unsampled_indices(
                        clf.random_state,
                        n_samples,
                        n_samples_bootstrap,
                    )
            else:
                indices = list(range(X.shape[0]))                    
        
            I = np.zeros((X.shape[0], clf.tree_.node_count))
            for j in indices:
                I[j,idx[j,i]] = 1.0    
            Is.append(I)
        return np.hstack(Is)  
    
    def fit(self,X,y,X_,y_,bias,sample_weight = None):
        if sample_weight is not None:
            self.forest.fit(
                            X.reshape(-1,X.shape[2]), y.flatten(), sample_weight.flatten()
                            )
        else:
            self.forest.fit(
                            X.reshape(-1,X.shape[2]), y.flatten() 
                            )
                
        
        I = self.getIndicators(self.forest, X.reshape(-1,X.shape[2]), False, False)        
        for _ in range(self.max_iter):
            self.network.hidden_layer_sizes = (I.shape[1],) + (I.shape[1],) + (self.network.hidden_layer_sizes[2],)
            self.network.dual_fit(X_, y_, I.reshape((y_.shape[0],y_.shape[1],-1)),
                                   bias = bias, par_lr = self.learning_rate,
                                   recurrent_hidden = 3)
            
            y_pred,_,I = self.predict_proba(X_, bias, returnI = True)
            
            
            del self.network
            self.network = copy.deepcopy(self.network_estimator)
    
    def predict_proba(self, X, bias, returnI = False ):
        res = np.zeros((X.shape[0],X.shape[1]))
        hidden = np.zeros((X.shape[0],X.shape[1],self.network._coeffs[0].shape[0]))
        I_list = []
        for t in range(X.shape[1]):
            if t > 0:
                X_aug = np.hstack([hidden[:, t - 1],X[:,t]])
            else:
                X_aug = np.hstack([hidden[:, 0],X[:,t]])    
                
            I = self.getIndicators(self.forest, X_aug, False, False)
            res[:,t], hidden[:,t] = self.network.predict_proba(I,  bias = bias, par_lr = self.learning_rate)
            I_list.append(I)
        if returnI:
            return res, hidden, np.swapaxes(np.asarray(I_list),0,1)
        else:    
            return res, hidden    
        