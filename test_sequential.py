'''
Created on 26 сент. 2023 г.

@author: keen
'''

from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from boosted_sequential import CascadeSequentialClassifier
from sklearn.metrics import log_loss
import itertools

import tensorflow as tf


#X = np.asarray([[0,0,1,1,0],[1,0,0,0,0],[0,1,0,0,1],[1,0,1,1,0],[1,1,0,0,1],[0,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]).reshape(8,5,1)
#y = np.asarray([[0,1,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,1,1,0,1],[1,0,0,1,1],[1,1,1,1,0],[0,0,0,0,0],[1,1,1,1,1]]).reshape(8,5)

def polyndrome(n):
    grids = []
    for k in range(1,n):
        which = np.array(list(itertools.combinations(range(n), k)))
        grid = np.zeros((len(which), 2*n), dtype="float")
        
        grid[np.arange(len(which))[None].T, which] = 1
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i,grid.shape[1] - j - 1] = grid[i,j]
        grids.append(grid)        
    return np.vstack(grids)       
    

X = polyndrome(10)

y = X[:,1:]
X = X[:,:-1] 
X = X.reshape(X.shape[0],X.shape[1],1)

print("Dataset: ", X.shape[0],X.shape[1])

x_train, x_validate, Y_train, Y_validate = train_test_split(
    X, y, test_size=0.5, shuffle=True
)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def ce_score(logits, labels):
    ce = log_loss(labels.flatten(),sigmoid(logits.reshape(-1, 1)))
    return  np.exp(ce / labels.shape[1])  

def make_model(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    output_layer = tf.keras.layers.SimpleRNN(2)(input_layer)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

epochs = 50
batch_size = 32

model = make_model(input_shape=x_train.shape[1:])
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = tf.keras.models.load_model("best_model.h5")

Y_v = model.predict(x_validate)


print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")

print("Cross-entropy: ", ce_score(Y_v, Y_validate))

model = CascadeSequentialClassifier(C=1.0, n_layers=5, verbose=2, n_estimators = 4, max_depth=5,max_features='sqrt')#, n_iter_no_change = 1, validation_fraction = 0.1)

model.fit(x_train, Y_train)#, monitor = monitor)

Y_v = model.predict(x_validate)


print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")

print("Cross-entropy: ", ce_score(Y_v, Y_validate))
