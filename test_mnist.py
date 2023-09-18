'''
Created on Sep 15, 2023

@author: keen
'''
from sklearn import preprocessing
from sklearn import datasets, metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from boosted_forest import CascadeBoostingClassifier

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

model = GradientBoostingClassifier(n_estimators=100)

model.fit(x_train, Y_train)

Y_v = model.predict(x_validate)

print(
    f"GDB Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)

model = CascadeBoostingClassifier()

model.fit(x_train, Y_train)

Y_v = model.predict(x_validate)

print(
    f"Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate, Y_v)}\n"
)






