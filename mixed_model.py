'''
Created on Jan 29, 2024

@author: keen
'''
import copy
import numpy as np
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices
from sklearn.metrics import accuracy_score
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy

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

class MixedModel:
    def __init__(self, forest_estimator, network_estimator, max_iter = 5,learning_rate = 1.):
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
                            X.reshape(-1,X.shape[2]), y.reshape(-1,y.shape[2]), sample_weight.flatten()
                            )
        else:
            self.forest.fit(
                            X.reshape(-1,X.shape[2]), y.reshape(-1,y.shape[2]) 
                            )
                
        #print("Forest train X: ",X[1,2,:20])
        #print("Importances: ", self.forest.feature_importances_)
        I = self.getIndicators(self.forest, X.reshape(-1,X.shape[2]), False, False)     
        
        #np.save("X.npy", X)
        #np.save("I.npy", I)   
        
        for i in range(self.max_iter):
            print ("Outer loop iter: ", i)
            self.network.hidden_layer_sizes = (I.shape[1],) + (I.shape[1],) + (self.network.hidden_layer_sizes[2],)
#             mm, hidden_grad = self.network.dual_fit(X_, y_, X, I.reshape((y_.shape[0],y_.shape[1],-1)),
#                                    bias = bias, par_lr = self.learning_rate,
#                                    recurrent_hidden = 3)
            
            mm, hidden_grad = self.network.mockup_fit(X_, y_, X, I.reshape((y_.shape[0],y_.shape[1],-1)),
                                   bias = bias, par_lr = self.learning_rate,
                                   recurrent_hidden = 3)            
            
            tmp = self.network
            self.network = mm
            y_pred,_,I_,X = self.predict_proba(X_, bias, returnI = True, learning_rate=self.learning_rate)
            
            #print("I diff: ", np.bitwise_xor(I.flatten().astype(int),I_.flatten().astype(int)).sum(), " of ", I.flatten().shape[0])
            I = I_
            self.network = tmp
            encoded_classes = np.asarray(y_pred.flatten() >= 0, dtype=int)
            
            print("Mixed score: ", accuracy_score(encoded_classes, y_.flatten()))
                    
            l = binary_log_loss(y_.flatten(), y_pred.flatten())
                                
            print("Mixed loss: ", l)                    
            
            r, _ = self.network.predict_proba(X_,  bias = bias, par_lr = self.learning_rate)
            encoded_classes = np.asarray(r.flatten() >= 0, dtype=int)
            
            print("Mixed score 2: ", accuracy_score(encoded_classes, y_.flatten()))            
            I = I.reshape((-1,I.shape[2])) 
        self.network = mm
        return hidden_grad
        
    def predict_proba(self, X, bias, returnI = False, learning_rate = 1.0):
        res = np.zeros((X.shape[0],X.shape[1]))
        hidden = np.zeros((X.shape[0],X.shape[1],self.network.coefs_[0].shape[1]))
        I_list = []
        X_augs = []
        for t in range(X.shape[1]):
            if t > 0:
                X_aug = np.hstack([hidden[:, t - 1],X[:,t]])
            else:
                X_aug = np.hstack([hidden[:, 0],X[:,t]])    
            
            X_augs.append(X_aug)
            I = self.getIndicators(self.forest, X_aug, False, False)
            I_list.append(I)
            I = np.swapaxes(np.asarray(I_list),0,1)
            res[:,:t+1], hidden[:,:t+1] = self.network.predict_proba(I,  bias = bias, par_lr = learning_rate)

        print("Mixed prediction result sample: ", res[0][:5]) 
        print("With bias: ", bias[0][1][:5])
        print("With X_augs: ", X_aug[0])
        
        if returnI:
            return res, hidden, I, np.swapaxes(np.asarray(X_augs),0,1)
        else:    
            return res, hidden    
        