'''
Created on 27 сент. 2023 г.

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
from sklearn.metrics import log_loss

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

from boosted_forest import _init_raw_predictions, BaseBoostedCascade


class BaseSequentialBoostingDummy(BaseBoostedCascade):
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
        
        def sigmoid(x):
            return 1. / (1 + np.exp(-x))
          

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
        
        history = np.zeros(raw_predictions.shape)

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
                history,
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
                print("Cross-entropy: ", log_loss(y.flatten(),sigmoid(raw_predictions.reshape(-1, 1))))

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
        history,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        assert sample_mask.dtype == bool
        loss = self._loss
        original_y = y
        
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
            n_estimators=100*(i+1)
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
            n_estimators=100*(i+1)
        )        

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()
        binner_ = Binner(
            n_bins=self.n_bins,
            bin_subsample=self.bin_subsample,
            bin_type=self.bin_type,
            random_state=self.random_state,
        )  
        
        self.binners.append(binner_)      
        
        rp_old = raw_predictions.copy()
        rp_old_bin = self._bin_data(binner_, rp_old, is_training_data=True)       
        
        if loss.n_classes == 2:
            residual = np.zeros((X.shape[0],X.shape[1], 1))
        else:    
            residual = np.zeros((X.shape[0],X.shape[1], loss.n_classes))     
            
        X = X_csr if X_csr is not None else X

        if  loss.n_classes == 2:
            K = 1
        else:
            K = loss.n_classes
        
        alpha = 0.1
        for t in reversed(range(0,X.shape[1])):
            #dummy loss
            neg_grad = - loss.gradient(
                y[:,t].copy(order='C'), raw_predictions_copy[:,t].copy(order='C') 
            )
            
            if not self.dummy_loss and i > 0 and t < X.shape[1]-1:
                if K > 1: 
                    for k in range(K):
                        neg_grad[:,k] -= alpha*np.multiply(history[:,t + 1,k], residual[:,t + 1])
                        for t2 in range(t + 1,X.shape[1]):
                            h = np.ones(shape=(history.shape[0],))
                            for t3 in range(t + 1,t2 + 1):
                                h = np.multiply(h,history[:,t3, k])
                            next_grad = - loss.gradient(
                                y[:,t2].copy(order='C'), raw_predictions_copy[:,t2].copy(order='C') 
                            )   
                            neg_grad[:,k] -= alpha*np.multiply(h,next_grad[:,k].flatten())                        
                else:  

                    for t2 in range(t + 1,X.shape[1]):
                        h = np.ones(shape=(history.shape[0],))
                        for t3 in range(t + 1,t2 + 1):
                            h = np.multiply(h,history[:,t3, 0])
                        next_grad = - loss.gradient(
                            y[:,t2].copy(order='C'), raw_predictions_copy[:,t2].copy(order='C') 
                        )   
                        neg_grad -= alpha*np.multiply(h,next_grad.flatten())
            
            if loss.n_classes == 2:
                neg_grad = neg_grad.reshape(-1,1)
            
            residual[:,t] = neg_grad   
        
        history.fill(0.)    
        for k in range(K):  
            if loss.n_classes > 2:
                y = np.array(original_y == k, dtype=np.float64)
                # induce regression forest on residuals
            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)
    
            self.estimators_[i, k] = []
                   
            X_aug = np.zeros((X.shape[0],X.shape[1], X.shape[2] + 2))
             
            for t in range(X.shape[1]):                      
                if t > 0:
                    if isinstance(X,np.ndarray):
                        X_aug[:,t] = np.hstack([X[:,t],rp_old_bin[:,t - 1,k].reshape(-1,1), rp_old_bin[:,t,k].reshape(-1,1)])#[:,k].reshape(-1,1)])
                    else:
                        X_aug[:,t] = hstack([X[:,t], csr_matrix(rp_old_bin[:,t - 1,k].reshape(-1,1)), csr_matrix(rp_old_bin[:,t,k].reshape(-1,1))])#[:,k].reshape(-1,1))])
                else:
                    if isinstance(X,np.ndarray):
                        X_aug[:,t] = np.hstack([X[:,t],np.zeros(rp_old_bin[:,t,k].shape).reshape(-1,1), rp_old_bin[:,t,k].reshape(-1,1)])#[:,k].reshape(-1,1)])
                    else:
                        X_aug[:,t] = hstack([X[:,t], csr_matrix(np.zeros(rp_old_bin[:,t,k].shape)).reshape(-1,1), csr_matrix(rp_old_bin[:,t,k]).reshape(-1,1)])#[:,k].reshape(-1,1))])
                                            
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
                 
                history[:,:,k] += kfold_estimator.fit(X_aug.reshape(-1,X_aug.shape[2]), residual[:,:, k].reshape(-1,1), y, raw_predictions, rp_old.reshape(-1,residual.shape[2]), k, sample_weight)
                #kfold_estimator.update_terminal_regions(X_aug, y, raw_predictions, k)
                self.estimators_[i, k].append(kfold_estimator)
                    
    
                    # add tree to ensemble
                
        return raw_predictions.reshape(raw_predictions_copy.shape)   
    
    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        X = self._bin_data(self.binners[0], X, False)
        self._check_initialized()
        raw = np.zeros(
             shape=(X.shape[0], X.shape[1], 1), dtype=np.float64
         )
        
        if isinstance(X,np.ndarray):
            X_aug = np.hstack([X,raw])         
        else:
            X_aug = hstack([X,csr_matrix(raw)])  
        
        #X = self.estimators_[0, 0][0].estimator_[0]._validate_X_predict(X_aug)
        
        if self.init_ == "zero":
            raw_predictions = np.zeros(
                shape=(X.shape[0], X.shape[1], self.n_trees_per_iteration_), dtype=np.float64
            )
        else:
            raw_predictions = _init_raw_predictions(
                X, self.init_, self._loss, is_classifier(self)
            )
        return raw_predictions.reshape(X.shape[0],X.shape[1],-1)    
    
    def predict_stage(self, i, X, raw_predictions):
        rp = self._bin_data(self.binners[i + 1], raw_predictions, False)
        new_raw_predictions = np.zeros(raw_predictions.shape)
        if self._loss.n_classes == 2:
            K = 1
        else:
            K = self._loss.n_classes    
        for t in range(X.shape[1]):
            for k in range(K):
                
                for estimator in self.estimators_[i,k]:
                    if t > 0:
                        if isinstance(X,np.ndarray):
                            X_aug = np.hstack([X[:,t],rp[:,t-1,k].reshape(-1,1), rp[:,t,k].reshape(-1,1)])         
                        else:
                            X_aug = hstack([X[:,t],csr_matrix(rp[:,t-1, k]).reshape(-1,1),csr_matrix(rp[:,t, k]).reshape(-1,1)])
                    else:
                        if isinstance(X,np.ndarray):
                            X_aug = np.hstack([X[:,t],np.zeros(rp[:,t,k].shape).reshape(-1,1), rp[:,t,k].reshape(-1,1)])#[:,k].reshape(-1,1)])
                        else:
                            X_aug = hstack([X[:,t], csr_matrix(np.zeros(rp[:,t,k].shape)).reshape(-1,1), csr_matrix(rp[:,t,k]).reshape(-1,1)])#[:,k].reshape(-1,1))])

                    new_raw_predictions[:,t,k] += estimator.predict(X_aug)
        raw_predictions += new_raw_predictions     
        
class CascadeSequentialClassifier(ClassifierMixin, BaseSequentialBoostingDummy):
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
        self.dummy_loss = False

    def _encode_y(self, y, sample_weight):
        # encode classes into 0 ... n_classes - 1 and sets attributes classes_
        # and n_trees_per_iteration_
        check_classification_targets(y)

        label_encoder = LabelEncoder()
        encoded_y_int = label_encoder.fit_transform(y.flatten())
        self.classes_ = label_encoder.classes_
        n_classes = self.classes_.shape[0]
        # only 1 tree for binary classification. For multiclass classification,
        # we build 1 tree per class.
        self.n_trees_per_iteration_ = 1 if n_classes <= 2 else n_classes
        encoded_y = encoded_y_int.astype(float, copy=False).reshape(y.shape)

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
        raw_predictions = self._raw_predict(X)
        if raw_predictions.shape[1] == 1:
            return raw_predictions.ravel()
        return raw_predictions

    def staged_decision_function(self, X):
        yield from self._staged_raw_predict(X)

    def predict(self, X):
        raw_predictions = self.decision_function(X)
        if self._loss.n_classes == 2:  # decision_function already squeezed it
            encoded_classes = (raw_predictions >= 0).astype(int)
        else:
            encoded_classes = np.argmax(raw_predictions, axis=2)
            
        return self.classes_[encoded_classes].reshape(X.shape[0],X.shape[1])


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
            
    def fit(self, X, y, sample_weight=None, monitor=None):
        if not self.warm_start:
            self._clear_state()

        # Check input
        # Since check_array converts both X and y to the same dtype, but the
        # trees use different types for X and y, checking them separately.

        sample_weight_is_none = sample_weight is None
        #sample_weight = _check_sample_weight(sample_weight, X)
        if sample_weight_is_none:
            y = self._encode_y(y=y, sample_weight=None)
        else:
            y = self._encode_y(y=y, sample_weight=sample_weight)
        self.n_features_in_ = X.shape[2] 
        self._set_max_features()

        # self.loss is guaranteed to be a string
        self._loss = self._get_loss(sample_weight=sample_weight)

        if self.n_iter_no_change is not None:
            stratify = y if is_classifier(self) else None
            (
                X_train,
                X_val,
                y_train,
                y_val,
                sample_weight_train,
                sample_weight_val,
            ) = train_test_split(
                X,
                y,
                sample_weight,
                random_state=self.random_state,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                if self.n_classes_ != np.unique(y_train).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        "The training data after the early stopping split "
                        "is missing some classes. Try using another random "
                        "seed."
                    )
        else:
            X_train, y_train, sample_weight_train = X, y, sample_weight
            X_val = y_val = sample_weight_val = None

        n_samples = X_train.shape[0]

        # First time calling fit.
        if not self._is_fitted():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == "zero":
                raw_predictions = np.zeros(
                    shape=(n_samples, self.n_trees_per_iteration_),
                    dtype=np.float64,
                )
            else:
                # XXX clean this once we have a support_sample_weight tag
                if sample_weight_is_none:
                    self.init_.fit(X_train.reshape(-1,X_train.shape[2]), y_train.flatten())
                else:
                    msg = (
                        "The initial estimator {} does not support sample "
                        "weights.".format(self.init_.__class__.__name__)
                    )
                    try:
                        self.init_.fit(
                            X_train.reshape(-1,X_train.shape[2]), y_train.flatten(), sample_weight=sample_weight_train
                        )
                    except TypeError as e:
                        if "unexpected keyword argument 'sample_weight'" in str(e):
                            # regular estimator without SW support
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise
                    except ValueError as e:
                        if (
                            "pass parameters to specific steps of "
                            "your pipeline using the "
                            "stepname__parameter"
                            in str(e)
                        ):  # pipeline
                            raise ValueError(msg) from e
                        else:  # regular estimator whose input checking failed
                            raise

                raw_predictions = _init_raw_predictions(
                    X_train, self.init_, self._loss, is_classifier(self)
                )
                
                raw_predictions = raw_predictions.reshape(X.shape[0],X.shape[1],-1)
 
            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        # warm start: this is not the first time fit was called
        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError(
                    "n_estimators=%d must be larger or equal to "
                    "estimators_.shape[0]=%d when "
                    "warm_start==True" % (self.n_estimators, self.estimators_.shape[0])
                )
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _raw_predict
            # are more constrained than fit. It accepts only CSR
            # matrices. Finite values have already been checked in _validate_data.
            X_train = check_array(
                X_train,
                dtype=DTYPE,
                order="C",
                accept_sparse="csr",
                force_all_finite=False,
            )
            raw_predictions = self._raw_predict(X_train)
            self._resize_state()

        # fit the boosting stages
        n_stages = self._fit_stages(
            X_train,
            y_train,
            raw_predictions,
            sample_weight_train,
            self._rng,
            X_val,
            y_val,
            sample_weight_val,
            begin_at_stage,
            monitor,
        )

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, "oob_improvement_"):
                # OOB scores were computed
                self.oob_improvement_ = self.oob_improvement_[:n_stages]
                self.oob_scores_ = self.oob_scores_[:n_stages]
                self.oob_score_ = self.oob_scores_[-1]
        self.n_estimators_ = n_stages
        return self            
           
                          

