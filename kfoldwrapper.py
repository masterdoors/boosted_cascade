"""
Implementation of the estimator wrapper to support customized base estimators.
"""

__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from mixed_model import MixedModel

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

class IndBuilder:
    def __init__(self, estimator, indicator):
        self.estimator = estimator
        self.indicator = indicator
        
    def getIds(self, X, sampled = True, do_sample = True):
        return self.indicator(self.estimator,X,sampled,do_sample)    

class KFoldWrapper(object):
    """
    A general wrapper for base estimators without the characteristic of
    out-of-bag (OOB) estimation.
    """

    def __init__(
        self,
        forest_estimator,
        network_estimator,
        n_splits,
        C=1.0,
        factor = 0.5,
        random_state=None,
        hidden_size = 2,
        hidden_activation = 'tanh',
        learning_rate = 1.0,
        verbose=1,
    ):
     
        # Parameters were already validated by upstream methods
        self.dummy_estimator_f = forest_estimator
        self.dummy_estimator_n = network_estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        # Internal container
        self.estimators_ = []
        self.C = C
        self.factor = factor 
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_


    def fit(self, X_,X, y, y_,history_k,sample_weight=None):
        self.lr = []
        self.estimators_ = []
        n_samples = X.shape[0]
        out = np.zeros((n_samples,  X.shape[1]))  # pre-allocate results
        hidden = np.zeros((n_samples,  X.shape[1], self.hidden_size))        
        #print("y:",y.min(), y.max())
        kf = KFold(n_splits=self.n_splits, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X):
            bias = history_k
            estimator = MixedModel(self.dummy_estimator_f, self.dummy_estimator_n, max_iter = 15,learning_rate = self.learning_rate)
            if sample_weight is not None:
                estimator.fit(X[train_index],y[train_index],X_[train_index],y_[train_index],bias[train_index],sample_weight[train_index])
            else:
                estimator.fit(X[train_index],y[train_index],X_[train_index],y_[train_index],bias[train_index])     
            self.estimators_.append(estimator)
            out_, hidden_ = estimator.predict_proba(X_[test_index],bias=bias[test_index],learning_rate = self.learning_rate)
            
            out[test_index] += out_
            hidden[test_index,:] += hidden_
            
        return self.factor * out, self.factor * hidden    

            
    def predict(self, X,history):
        n_samples = X.shape[0]
        out = np.zeros((n_samples,  X.shape[1]))  # pre-allocate results
        hidden = np.zeros((n_samples,  X.shape[1], self.hidden_size))   
        for estimator in self.estimators_:
            out_, hidden_ = estimator.predict_proba(X,bias=history,learning_rate = self.learning_rate)  # classification
            out += out_
            hidden += hidden_            

        return self.factor * out / self.n_splits, self.factor * hidden / self.n_splits  # return the average prediction
