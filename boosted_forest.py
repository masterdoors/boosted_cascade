'''
Created on Sep 16, 2023

@author: keen
'''

from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._gb import VerboseReporter
from sklearn.dummy import DummyClassifier, DummyRegressor
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse import hstack
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.tree._tree import DOUBLE, DTYPE
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier

from kfoldwrapper import KFoldWrapper
from sklearn.ensemble import RandomForestRegressor
from numbers import Integral, Real
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import  accuracy_score
import time

from sklearn.ensemble import ExtraTreesRegressor
from _binner import Binner
from sklearn.model_selection import train_test_split

from sklearn._loss.loss import (
    _LOSSES,
    AbsoluteError,
    ExponentialLoss,
    HalfBinomialLoss,
    HalfMultinomialLoss,
    HalfSquaredError,
    HuberLoss,
    PinballLoss,
)


def _init_raw_predictions(X, estimator, loss, use_predict_proba):
    # TODO: Use loss.fit_intercept_only where appropriate instead of
    # DummyRegressor which is the default given by the `init` parameter,
    # see also _init_state.
    if len(X.shape) > 2:
        X_ = X.reshape(-1,X.shape[2])
    if use_predict_proba:
        # Our parameter validation, set via _fit_context and _parameter_constraints
        # already guarantees that estimator has a predict_proba method.
        predictions = estimator.predict_proba(X_)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # probability of positive class
        eps = np.finfo(np.float32).eps  # FIXME: This is quite large!
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X_).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)
    
class BaseBoostedCascade(BaseGradientBoosting):
    def _bin_data(self, binner, X, is_training_data=True):
        """
        Bin data X. If X is training data, the bin mapper is fitted first."""
        description = "training" if is_training_data else "testing"

        tic = time.time()
        if len(X.shape) > 2:
            X_ = X.reshape(-1,X.shape[2])
        else:
            X_ = X    
        
        if is_training_data:
            X_binned = binner.fit_transform(X_)
        else:
            X_binned = binner.transform(X_)
            X_binned = np.ascontiguousarray(X_binned)
        toc = time.time()
        binning_time = toc - tic

        if self.verbose > 1:
            msg = (
                "{} Binning {} data: {:.3f} MB => {:.3f} MB |"
                " Elapsed = {:.3f} s"
            )
            print(
                msg.format(
                    str(time.time()),
                    description,
                    X.nbytes / (1024 * 1024),
                    X_binned.nbytes / (1024 * 1024),
                    binning_time,
                )
            )
        if len(X.shape) > 2:
            X_binned = X_binned.reshape(X.shape) 
        return X_binned    
    
    def __init__(self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=2,
        n_layers=3,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        C=1.0,
        n_splits=5,
        n_bins=255,
        bin_subsample=200000,
        bin_type="percentile"):
        super().__init__(loss = loss,
                         learning_rate = learning_rate,
                         n_estimators = n_layers,
                         subsample = subsample,
                         criterion = criterion,
                         min_samples_split = min_samples_split,
                         min_samples_leaf = min_samples_leaf,
                         min_weight_fraction_leaf = min_weight_fraction_leaf,
                         max_depth = max_depth,
                         min_impurity_decrease = min_impurity_decrease,
                         init = init,
                         random_state = random_state,
                         max_features = max_features,
                         verbose = verbose,
                         max_leaf_nodes = max_leaf_nodes,
                         warm_start = warm_start,
                         validation_fraction = validation_fraction,
                         n_iter_no_change = n_iter_no_change,
                         tol = tol,
                         ccp_alpha = ccp_alpha
                         )
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.C = C
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.bin_subsample = bin_subsample
        self.bin_type = bin_type
        self.binners = []
        
    def _init_state(self):
        """Initialize model state and allocate model state data structures."""

        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy="prior")
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy="quantile", quantile=self.alpha)
            else:
                self.init_ = DummyRegressor(strategy="mean")

        self.estimators_ = np.empty(
            (self.n_layers, self.n_trees_per_iteration_), dtype=object
        )
        self.train_score_ = np.zeros((self.n_layers,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_layers), dtype=np.float64)
            self.oob_scores_ = np.zeros((self.n_layers), dtype=np.float64)
            self.oob_score_ = np.nan           
    
    def _fit_stages(
        self,
        X,
        y,
        raw_predictions,
        sample_weight,
        random_state,
        X_val,
        y_val,
        sample_weight_val,
        begin_at_stage=0,
        monitor=None,
    ):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )  
        
        self.binners.append(binner_)      
        X_ = self._bin_data(binner_, X, is_training_data=True)
        
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X_) if issparse(X) else None
        X_csr = csr_matrix(X_) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val, check_input=False)

        # perform boosting iterations
        i = begin_at_stage

        for i in range(begin_at_stage, self.n_layers):
            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag, random_state)
                if i == 0:  # store the initial loss to compute the OOB score
                    initial_loss = loss_(
                        y[~sample_mask],
                        raw_predictions[~sample_mask],
                        sample_weight[~sample_mask],
                    )

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i,
                X_,
                y,
                raw_predictions,
                sample_weight,
                sample_mask,
                random_state,
                X_csc,
                X_csr
            )
            
            # track loss
            if do_oob:
                self.train_score_[i] = loss_(
                    y[sample_mask],
                    raw_predictions[sample_mask],
                    sample_weight[sample_mask],
                )
                self.oob_scores_[i] = loss_(
                    y[~sample_mask],
                    raw_predictions[~sample_mask],
                    sample_weight[~sample_mask],
                )
                previous_loss = initial_loss if i == 0 else self.oob_scores_[i - 1]
                self.oob_improvement_[i] = previous_loss - self.oob_scores_[i]
                self.oob_score_ = self.oob_scores_[-1]
            else:
                # no need to fancy index w/ no subsampling
                if self._loss.n_classes == 2:
                    K = 1
                else:
                    K = self._loss.n_classes    
                self.train_score_[i] = loss_(y.flatten(), raw_predictions.reshape(-1, K), sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)
                if self._loss.n_classes == 2:
                    encoded_classes = np.asarray(raw_predictions.reshape(X.shape[0],X.shape[1]) >= 0, dtype=int)
                else:  
                    K = self._loss.n_classes    
                    encoded_classes = np.argmax(raw_predictions.reshape(X.shape[0],X.shape[1], K), axis=2)
                print("Acc: ",accuracy_score(encoded_classes.flatten(),y.flatten())) 

            if monitor is not None:
                if monitor(i, self, locals()):
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                next_ = next(y_val_pred_iter)
                validation_loss = loss_(y_val, next_, sample_weight_val)
                encoded_classes = np.argmax(next_, axis=1)
                print("val loss: ", validation_loss, "val_acc: ", accuracy_score(encoded_classes,y_val))

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break
          
        self.n_layers = i
        return i + 1   
    
    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of ``_n_classes`` trees to the boosting model."""

        assert sample_mask.dtype == bool
        loss = self._loss
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()
        
        estimator = RandomForestRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            oob_score = True,
            bootstrap=True,
            n_estimators=100
        )  
        
        restimator = ExtraTreesRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
            oob_score = True,
            bootstrap=True,
            n_estimators=100
        )        

        residual = - loss.gradient(
            y, raw_predictions_copy 
        )
        
        if len(residual.shape) == 1:
            residual = residual.reshape(-1,1)
            
        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )  
        
        self.binners.append(binner_)      
        
        rp_old = raw_predictions.copy()
        rp_old_bin = self._bin_data(binner_, rp_old, is_training_data=True)           
         
        for k in range(loss.n_classes):
            if loss.n_classes > 2:
                y = np.array(original_y == k, dtype=np.float64)

            # induce regression forest on residuals
            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            self.estimators_[i, k] = []
            X = X_csr if X_csr is not None else X
            
            if isinstance(X,np.ndarray):
                X_aug = np.hstack([X,rp_old_bin[:,k].reshape(-1,1)])
            else:
                X_aug = hstack([X, csr_matrix(rp_old_bin[:,k].reshape(-1,1))])               
            
            for eid  in range(self.n_estimators):
                if eid %2 == 0:
                    kfold_estimator = KFoldWrapper(
                        estimator,
                        self.n_splits,
                        self.C,
                        1. / self.n_estimators,
                        self.random_state,
                        self.verbose
                    )
                else:
                    kfold_estimator = KFoldWrapper(
                        restimator,
                        self.n_splits,
                        self.C,
                        1. / self.n_estimators,
                        self.random_state,
                        self.verbose
                    )                       
                 
                kfold_estimator.fit(X_aug, residual[:, k], y, raw_predictions, rp_old, k, sample_weight)
                #kfold_estimator.update_terminal_regions(X_aug, y, raw_predictions, k)
                self.estimators_[i, k].append(kfold_estimator)
                

                # add tree to ensemble
                
        return raw_predictions    
    
    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        X = self._bin_data(self.binners[0], X, False)
        self._check_initialized()
        raw = np.zeros(
             shape=(X.shape[0], 1), dtype=np.float64
         )
        
        if isinstance(X,np.ndarray):
            X_aug = np.hstack([X,raw])         
        else:
            X_aug = hstack([X,csr_matrix(raw)])  
        
        X = self.estimators_[0, 0][0].estimator_[0]._validate_X_predict(X_aug)
        
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, is_classifier(self)
            )
        return raw_predictions    
    
    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        self.predict_stages(X,raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        if check_input:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
        X = self._bin_data(self.binners[0], X, False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            self.predict_stage(i, X, raw_predictions)
            yield raw_predictions.copy() 
            
    def predict_stage(self, i, X, raw_predictions):
        rp = self._bin_data(self.binners[i + 1], raw_predictions, False)
        new_raw_predictions = np.zeros(raw_predictions.shape)
        for k in range(self._loss.n_classes):
            
            for estimator in self.estimators_[i,k]:
                if isinstance(X,np.ndarray):
                    X_aug = np.hstack([X,rp[:,k].reshape(-1,1)])         
                else:
                    X_aug = hstack([X,csr_matrix(rp[:, k]).reshape(-1,1)])  
                    
                new_raw_predictions[:,k] += estimator.predict(X_aug)
        raw_predictions += new_raw_predictions        
                    
    
    def predict_stages(self, X, raw_predictions):
        X = self._bin_data(self.binners[0], X, False)        
        for i in range(self.n_layers):
            self.predict_stage(i, X, raw_predictions)
 
class CascadeBoostingClassifier(ClassifierMixin, BaseBoostedCascade):
    _parameter_constraints: dict = {
        **BaseBoostedCascade._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=2,
        n_layers=3,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        C=1.0,
        init=None,
        random_state=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_layers=n_layers,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
            C=C,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )

    def _encode_y(self, y, sample_weight):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False)

        # From here on, it is additional to the HGBT case.
        # expose n_classes_ attribute
        self.n_classes_ = n_classes
        if sample_weight is None:
            n_trim_classes = n_classes
        else:
            n_trim_classes = np.count_nonzero(np.bincount(encoded_y_int, sample_weight))

        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        return encoded_y
    
    def _get_loss(self, sample_weight):
        if self.loss == "log_loss":
            if self.n_classes_ == 2:
                return HalfBinomialLoss(sample_weight=sample_weight)
            else:
                return HalfMultinomialLoss(
                    sample_weight=sample_weight, n_classes=self.n_classes_
                )
        elif self.loss == "exponential":
            if self.n_classes_ > 2:
                raise ValueError(
                    f"loss='{self.loss}' is only suitable for a binary classification "
                    f"problem, you have n_classes={self.n_classes_}. "
                    "Please use loss='log_loss' instead."
                )
            else:
                return ExponentialLoss(sample_weight=sample_weight) 
            
    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError(
                "y contains %d class after sample_weight "
                "trimmed classes with zero weights, while a "
                "minimum of 2 classes are required." % n_trim_classes
            )
        self._n_classes = len(self.classes_)
        # expose n_classes_ attribute
        self.n_classes_ = self._n_classes
        return y

    def decision_function(self, X):
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        if raw_predictions.ndim == 1:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=1)
            
        return self.classes_[encoded_classes]


    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            encoded_labels = self._loss._raw_prediction_to_decision(raw_predictions)
            yield self.classes_.take(encoded_labels, axis=0)

    def predict_proba(self, X):
        raw_predictions = self.decision_function(X)
        try:
            return self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)

    def staged_predict_proba(self, X):
        try:
            for raw_predictions in self._staged_raw_predict(X):
                yield self._loss._raw_prediction_to_proba(raw_predictions)
        except NotFittedError:
            raise
        except AttributeError as e:
            raise AttributeError(
                "loss=%r does not support predict_proba" % self.loss
            ) from e


class CascadeBoostingRegressor(RegressorMixin, BaseBoostedCascade):
    _parameter_constraints: dict = {
        **BaseBoostedCascade._parameter_constraints,
        "loss": [StrOptions({"squared_error", "absolute_error", "huber", "quantile"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict"])],
        "alpha": [Interval(Real, 0.0, 1.0, closed="neither")],
    }

    def __init__(
        self,
        *,
        loss="squared_error",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        min_impurity_decrease=0.0,
        init=None,
        C = 1.0,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state,
            C = C,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
        )

    def _validate_y(self, y, sample_weight=None):
        if y.dtype.kind == "O":
            y = y.astype(DOUBLE)
        return y
    
    def predict(self, X):
        X = self._validate_data(
            X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
        )
        # In regression we can directly return the raw value from the trees.
        return self._raw_predict(X).ravel()

    def staged_predict(self, X):
        for raw_predictions in self._staged_raw_predict(X):
            yield raw_predictions.ravel()

    def apply(self, X):
        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves   
    
