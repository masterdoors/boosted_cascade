"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
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

    def fit(self, X, y, y_, raw_predictions,rp_old,k,sample_weight=None):
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        self.lr = []
        for i, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            estimator = copy.deepcopy(self.dummy_estimator_)

            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_idx], y[train_idx])
            else:
                estimator.fit(
                    X[train_idx], y[train_idx], sample_weight[train_idx]
                )
                
            self.update_terminal_regions(estimator, X, y_, raw_predictions, rp_old,sample_weight,i, k, train_idx, val_idx) 
            
            self.estimators_.append(estimator)  
            
    def getIndicators(self, estimator, X):
        Is = []
        idx = estimator.apply(X)
        for i,clf in enumerate(estimator.estimators_):
            I = np.zeros((X.shape[0], clf.tree_.node_count))
            for j in range(X.shape[0]):
                I[j,idx[j,i]] = 1.0    
            Is.append(I)
        return np.hstack(Is)            
                    

    def update_terminal_regions(self,e, X, y,raw_predictions, rp_old, sample_weight, i, k, train_idx, val_idx):
        bias = rp_old[train_idx,k]
        self.lr.append(LogisticRegression(C=self.C,
                                fit_intercept=False,
                                solver='lbfgs',
                                max_iter=100,
                                multi_class='ovr', n_jobs=1))            
        
        I = self.getIndicators(e, X[train_idx])
  
        self.lr[i].fit(I, y[train_idx], bias = bias, sample_weight = sample_weight[train_idx])
        I = self.getIndicators(e, X[val_idx])
        raw_predictions[val_idx,k] += self.factor*self.lr[i].decision_function(I) 
    
    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, ))  # pre-allocate results
        for i, estimator in enumerate(self.estimators_):
            I = self.getIndicators(estimator, X)
            out += self.lr[i].decision_function(I)  # classification
        return self.factor * out / self.n_splits  # return the average prediction
