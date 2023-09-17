"""
Implementation of the estimator wrapper to support customized base estimators.
"""


__all__ = ["KFoldWrapper"]

import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


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
        n_outputs,
        C=1.0,
        random_state=None,
        verbose=1,
        is_classifier=True,
        parallel=False
    ):

        # Parameters were already validated by upstream methods
        self.dummy_estimator_ = estimator
        self.n_splits = n_splits
        self.n_outputs = n_outputs
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        # Internal container
        self.estimators_ = []
        self.parallel = parallel
        self.C = C

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_

    def fit(self, X, y, raw_predictions, sample_weight=None):
        n_samples, _ = X.shape
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for k, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            y_ = (y - raw_predictions[:,k]).astype(int)
            estimator = copy.deepcopy(self.dummy_estimator_)

            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_idx], y_[train_idx])
            else:
                estimator.fit(
                    X[train_idx], y_[train_idx], sample_weight[train_idx]
                )
            self.estimators_.append(estimator)    

    def update_terminal_regions(self,X, y,raw_predictions, k):
        preds = []
        self.lr = []
        for i,e in enumerate(self.estimators_):
            self.lr.append(LogisticRegression(C=self.C,
                                    fit_intercept=False,
                                    solver='lbfgs',
                                    max_iter=100,
                                    multi_class='multinomial', n_jobs=-1))            
            
            I = e.apply(X)
      
            self.lr[i].fit(I, y)
            preds.append(self.lr[i].decision_function(I))             
        raw_predictions[:,k] += preds.mean(axis=0)
                

    def predict(self, X):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, self.n_outputs))  # pre-allocate results
        for estimator in self.estimators_:
            if self.is_classifier:
                out += estimator.apply(X)  # classification
            else:
                if self.n_outputs > 1:
                    out += estimator.apply(X)  # multi-variate regression
                else:
                    out += estimator.apply(X).reshape(
                        n_samples, -1
                    )  # univariate regression
        I = np.hstack(out)
        return self.lr.decision_function(I)  # return the average prediction
