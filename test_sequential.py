'''
Created on 26 сент. 2023 г.

@author: keen
'''

from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from boosted_sequential import CascadeSequentialClassifier



X = np.asarray([[0,0,1,1,0],[1,0,0,0,0],[0,1,0,0,1],[1,0,1,1,0],[1,1,0,0,1],[0,1,1,1,1]]).reshape(5,6,1)
y = np.asarray([[0,1,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,1,1,0,1],[1,0,0,1,1],[1,1,1,1,0]]).reshape(5,6)

x_train, x_validate, Y_train, Y_validate = train_test_split(
    X, y, test_size=0.5, shuffle=True
)

model = CascadeSequentialClassifier(C=1.0, n_layers=5, verbose=2, n_estimators = 4, max_depth=5,max_features='sqrt')#, n_iter_no_change = 1, validation_fraction = 0.1)

model.fit(x_train, Y_train)#, monitor = monitor)

Y_v = model.predict(x_validate)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")
