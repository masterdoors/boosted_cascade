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
from biased_mlp import BiasedMLPClassifier
from sklearn.metrics import accuracy_score

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
        hidden_size = 2,
        hidden_activation = 'tanh',
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
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation

    @property
    def estimator_(self):
        """Return the list of internal estimators."""
        return self.estimators_


    def fit(self, X, y, y_,history_k,sample_weight=None):
        estimator = copy.deepcopy(self.dummy_estimator_)
        self.lr = []
        self.estimators_ = []
        n_samples, _ = X.shape
        out = np.zeros((n_samples, ))  # pre-allocate results
        hidden = np.zeros((n_samples, self.hidden_size))        
        
        kf = KFold(n_splits=self.n_splits, random_state=None, shuffle=True)
        for i,(train_index, test_index) in enumerate(kf.split(X)):
            estimator = copy.deepcopy(self.dummy_estimator_)
            # Fit on training samples
            if sample_weight is None:
                # Notice that a bunch of base estimators do not take
                # `sample_weight` as a valid input.
                estimator.fit(X[train_index], y[train_index].flatten())
            else:
                estimator.fit(
                    X[train_index], y[train_index].flatten(), sample_weight[train_index]
                )
             
            bias = history_k
            self.lr.append(BiasedMLPClassifier(alpha=1./self.C,hidden_layer_sizes=self.hidden_size,
                                               activation=self.hidden_activation,verbose=False,
                                                max_iter=10000,
                                                learning_rate_init=0.0001))            
            
            I_ = self.getIndicators(estimator, X[train_index], False, False)#False)
            self.lr[i].fit(I_, y_.flatten()[train_index], bias = bias[train_index])#, sample_weight = sample_weight)
            out_, hidden_ = self.lr[i].predict_proba(I_,bias=bias[train_index])
            out_ = np.asarray(out_ >= 0, dtype=int).flatten()
            s1 = accuracy_score(out_,y_.flatten()[train_index])
            self.estimators_.append(estimator)
            #TODO add raw predictions again
            I_ = self.getIndicators(estimator, X[test_index], False, False)
            out_, hidden_ = self.lr[i].predict_proba(I_,bias=bias[test_index])
            
            out[test_index] += out_
            hidden[test_index,:] += hidden_
            out_ = np.asarray(out_ >= 0, dtype=int).flatten()
            print("KF acc:", s1, accuracy_score(out_,y_.flatten()[test_index]))            
            
        return self.factor * out, self.factor * hidden     
            
            
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
                    

    def predict(self, X,history):
        n_samples, _ = X.shape
        out = np.zeros((n_samples, ))  # pre-allocate results
        hidden = np.zeros((n_samples, self.hidden_size))
        for i, estimator in enumerate(self.estimators_):
            I = self.getIndicators(estimator, X, do_sample = False)
            out_, hidden_ = self.lr[i].predict_proba(I,bias=history)  # classification
            out += out_
            hidden += hidden_            

        return self.factor * out / self.n_splits, self.factor * hidden / self.n_splits  # return the average prediction
