'''
Created on Sep 15, 2023

@author: keen
'''
from sklearn import preprocessing
from sklearn import datasets, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from deepforest import CascadeForestClassifier
from boosted_forest import CascadeBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))

Y =  np.asarray(digits.target).astype('int64')

print (np.unique(Y,return_counts=True))

for i in range(len(Y)):
    Y[i] = Y[i] + 1

x = preprocessing.normalize(data, copy=False, axis = 0)

x_train, x_validate, Y_train, Y_validate = train_test_split(
    x, Y, test_size=0.5, shuffle=False
)

def monitor(i, e, locals):
    if i > 0 and ((e.train_score_[i] > e.train_score_[i - 1]) or e.train_score_[i] < 0.0001):
        return True
    else: 
        return False

model = GradientBoostingClassifier(n_estimators=100, verbose=2, n_iter_no_change = 1)

model.fit(x_train, Y_train, monitor = monitor)

Y_v = model.predict(x_validate)

print(
    f"GDB Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)

model = CascadeBoostingClassifier(C=1.0, n_layers=10, verbose=2, n_estimators = 4, max_depth=4, n_iter_no_change = 1)

model.fit(x_train, Y_train, monitor = monitor)

Y_v = model.predict(x_validate)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)

model = CascadeForestClassifier(backend='sklearn')
#est = [RandomForestClassifier(max_depth=None), ExtraTreesClassifier(max_depth=None),RandomForestClassifier(max_depth=None), ExtraTreesClassifier(max_depth=None)]                            
#model.set_estimator(est) 


model.fit(x_train, Y_train)

Y_v = model.predict(x_validate)

print(
    f"Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)





