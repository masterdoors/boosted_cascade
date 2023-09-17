'''
Created on Sep 16, 2023

@author: keen
'''

from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._gb import VerboseReporter
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.sparse import hstack
from sklearn.ensemble._gradient_boosting import _random_sample_mask
from sklearn.tree._tree import DOUBLE, DTYPE
from _boosted_cascade import predict_stages, predict_stage
from kfoldwrapper import KFoldWrapper
from sklearn.ensemble import RandomForestRegressor


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
        C=1.0):
        super().__init__(loss = loss,
                         learning_rate = learning_rate,
                         n_estimators = n_estimators,
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
        self.C = C
    
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
            splitter="best",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            ccp_alpha=self.ccp_alpha,
        )  
        
        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(
                y, raw_predictions_copy, k=k, sample_weight=sample_weight
            )

            # induce regression forest on residuals


            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            self.estimators_[i, k] = []
            for _, estimator in enumerate(self.n_estimators):
                kfold_estimator = KFoldWrapper(
                    estimator,
                    self.n_splits,
                    self.n_outputs,
                    self.C,
                    self.random_state,
                    self.verbose,
                    self.is_classifier, 
                    self.parallel
                )
    
                X = X_csr if X_csr is not None else X
                kfold_estimator.fit(X, residual, sample_weight)
                
                
                kfold_estimator.update_terminal_regions(X, y, raw_predictions, k)
                # add tree to ensemble
                self.estimators_[i, k].append(kfold_estimator)

        return raw_predictions    
    
    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        self.predict_stages(X,raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        """Compute raw predictions of ``X`` for each iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each stage.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            If False, the input arrays X will not be checked.

        Returns
        -------
        raw_predictions : generator of ndarray of shape (n_samples, k)
            The raw predictions of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
            Regression and binary classification are special cases with
            ``k == 1``, otherwise ``k==n_classes``.
        """
        if check_input:
            X = self._validate_data(
                X, dtype=DTYPE, order="C", accept_sparse="csr", reset=False
            )
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            self.predict_stage(i, X, raw_predictions)
            yield raw_predictions.copy() 
            
    def predict_stage(self, i, X, raw_predictions):
        for k in range(self._loss.K):
            for estimator in self.estimators_[i,k]:
                raw_predictions[:,k] += estimator.predict(X)    
    
    def predict_stages(self, X, raw_predictions):
        for i in range(self.n_layers):
            self.predict_stage(i, X, raw_predictions)
                        
