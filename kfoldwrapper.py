"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble._forest import _generate_unsampled_indices, _get_n_samples_bootstrap, _generate_sample_indices
from _logistic import LogisticRegression


from joblib import Parallel, delayed

def kfoldtrain(k,X,y,train_idx, dummy_estimator_,sample_weight):
    estimator = copy.deepcopy(dummy_estimator_)

    # Fit on training samples
    if sample_weight is None:
        # Notice that a bunch of base estimators do not take
        # `sample_weight` as a valid input.
        estimator.fit(X[train_idx], y[train_idx])
    else:
        estimator.fit(
            X[train_idx], y[train_idx], sample_weight[train_idx]
        )

    return k,estimator

class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    """

    def __init__(
        self,
        estimator,
        n_splits,
        C=1.0,
        factor = 0.5,
        random_state=None,
        verbose=1,
    ):
     
        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        # Internal container
        self.estimators_ = []
        self.C = C
        self.factor = factor 

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

#     def fit(self, X, y, y_, raw_predictions,rp_old,k,sample_weight=None):
#         splitter = KFold(
#             n_splits=self.n_splits,
#             shuffle=True,
#             random_state=self.random_state,
#         )
#         self.lr = []
#         for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
#             estimator = copy.deepcopy(self.dummy_estimator_)
# 
#             # Fit on training samples
#             if sample_weight is None:
#                 # Notice that a bunch of base estimators do not take
#                 # `sample_weight` as a valid input.
#                 estimator.fit(X[train_idx], y[train_idx])
#             else:
#                 estimator.fit(
#                     X[train_idx], y[train_idx], sample_weight[train_idx]
#                 )
#                 
#             self.update_terminal_regions(estimator, X, y_, raw_predictions, rp_old,sample_weight,i, k, train_idx, val_idx) 
#             
#             self.estimators_.append(estimator) 
            
            
    def fit(self, X, y, y_, raw_predictions,rp_old,k,sample_weight=None):
        estimator = copy.deepcopy(self.dummy_estimator_)
        self.lr = []
        # Fit on training samples
        if sample_weight is None:
            # Notice that a bunch of base estimators do not take
            # `sample_weight` as a valid input.
            estimator.fit(X, y)
        else:
            estimator.fit(
                X, y, sample_weight
            )
            
        history = self.update_terminal_regions(estimator, X, y_, raw_predictions, rp_old,sample_weight, k) 
        
        self.estimators_.append(estimator)
        return history  
            
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
                    

    def update_terminal_regions(self,e, X, y,raw_predictions, rp_old, sample_weight, k):
        bias = rp_old[:,k]
        self.lr.append(LogisticRegression(C=self.C,
                                fit_intercept=False,
                                solver='lbfgs',
                                max_iter=1000,
                                multi_class='ovr', n_jobs=-1))            
        
        I = self.getIndicators(e, X, False)
  
        self.lr[0].fit(I, y.flatten(), bias = bias, sample_weight = sample_weight)
        I = self.getIndicators(e, X,True)
        if len(raw_predictions.shape) == 2:
            history = self.factor*self.lr[0].decision_function(I)
            raw_predictions[:,k] += history
        else:
            history = self.factor*self.lr[0].decision_function(I).reshape(raw_predictions.shape[0],raw_predictions.shape[1])
            raw_predictions[:,:,k] += history
        return history    
                 
    
    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, ))  # pre-allocate results
        for i, estimator in enumerate(self.estimators_):
            I = self.getIndicators(estimator, X, do_sample = False)
            out += self.lr[i].decision_function(I)  # classification
        return self.factor * out #/ self.n_splits  # return the average prediction
