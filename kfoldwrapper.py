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

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

    def fit(self, X, y, raw_predictions, k, sample_weight=None):
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        
        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
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
            self.estimators_.append(estimator)    

    def update_terminal_regions(self,X, y,raw_predictions, k):
        preds = []
        self.lr = []
        y_ = y.copy()
        y_[y_ == 0] = -1
        margin = y_ - raw_predictions[:,k]
        y_[margin < 0] = -1
        y_[margin >= 0] = 1  
        
        for i,e in enumerate(self.estimators_):
            self.lr.append(LogisticRegression(C=self.C,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1))            
            
            I = e.apply(X)
      
            self.lr[i].fit(I, y_)
            preds.append(self.lr[i].decision_function(I))             
        raw_predictions[:,k] += np.asarray(preds).mean(axis=0)
                

    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, 1))  # pre-allocate results
        for i, estimator in enumerate(self.estimators_):
            out += self.lr[i].decision_function(estimator.apply(X))  # classification
        return out / self.n_splits  # return the average prediction
