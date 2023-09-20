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
from sklearn.exceptions import NotFittedError
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier

from kfoldwrapper import KFoldWrapper
from sklearn.ensemble import RandomForestRegressor
from numbers import Integral, Real
from sklearn.preprocessing import LabelEncoder
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
    """Return the initial raw predictions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data array.
    estimator : object
        The estimator to use to compute the predictions.
    loss : BaseLoss
        An instace of a loss function class.
    use_predict_proba : bool
        Whether estimator.predict_proba is used instead of estimator.predict.

    Returns
    -------
    raw_predictions : ndarray of shape (n_samples, K)
        The initial raw predictions. K is equal to 1 for binary
        classification and regression, and equal to the number of classes
        for multiclass classification. ``raw_predictions`` is casted
        into float64.
    """
    # TODO: Use loss.fit_intercept_only where appropriate instead of
    # DummyRegressor which is the default given by the `init` parameter,
    # see also _init_state.
    if use_predict_proba:
        # Our parameter validation, set via _fit_context and _parameter_constraints
        # already guarantees that estimator has a predict_proba method.
        predictions = estimator.predict_proba(X)
        if not loss.is_multiclass:
            predictions = predictions[:, 1]  # probability of positive class
        eps = np.finfo(np.float32).eps  # FIXME: This is quite large!
        predictions = np.clip(predictions, eps, 1 - eps, dtype=np.float64)
    else:
        predictions = estimator.predict(X).astype(np.float64)

    if predictions.ndim == 1:
        return loss.link.link(predictions).reshape(-1, 1)
    else:
        return loss.link.link(predictions)

class BaseBoostedCascade(BaseGradientBoosting):
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
        n_splits=3):
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
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples,), dtype=bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self._loss

        if self.verbose:
            verbose_reporter = VerboseReporter(verbose=self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None
        X_csc_aug = None
        X_csr_aug = None

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

            #if isinstance(X,np.ndarray):
            #    X_aug = np.hstack([X,raw_predictions])
            #else:
            #    if isinstance(X,csc_matrix):
            #        X_aug = hstack([X, csc_matrix(raw_predictions)])
            #    else:
            #        X_aug = hstack([X, csr_matrix(raw_predictions)])    
                    
            #if X_csc is not None:
            #    X_csc_aug = hstack([X_csc, csc_matrix(raw_predictions)])
            #    X_csr_aug = hstack([X_csr, csr_matrix(raw_predictions)])
            
            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i,
                X,#X_aug,
                y,
                raw_predictions,
                sample_weight,
                sample_mask,
                random_state,
                X_csc,#X_csc_aug,
                X_csr#X_csr_aug,
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
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter), sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

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
            n_estimators=10
        )  
        
        for k in range(loss.n_classes):
            if loss.n_classes > 2:
                y = np.array(original_y == k, dtype=np.float64)

            residual = - loss.gradient(
                y, raw_predictions_copy 
            )

            # induce regression forest on residuals


            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            self.estimators_[i, k] = []
            for _  in range(self.n_estimators):
                kfold_estimator = KFoldWrapper(
                    estimator,
                    self.n_splits,
                    self.C,
                    self.random_state,
                    self.verbose
                )
    
                X = X_csr if X_csr is not None else X
                kfold_estimator.fit(X, residual, raw_predictions, k, sample_weight)
                
                
                kfold_estimator.update_terminal_regions(X, y, raw_predictions, k)
                # add tree to ensemble
                self.estimators_[i, k].append(kfold_estimator)

        return raw_predictions    
    
    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0][0].estimator_[0]._validate_X_predict(X)
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
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            self.predict_stage(i, X, raw_predictions)
            yield raw_predictions.copy() 
            
    def predict_stage(self, i, X, raw_predictions):
        for k in range(self._loss.n_classes):
            for estimator in self.estimators_[i,k]:
                raw_predictions[:,k] += estimator.predict(X)    
    
    def predict_stages(self, X, raw_predictions):
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
