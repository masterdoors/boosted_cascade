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

from formal_models.pcfg_length_sampling import LengthSampler
from formal_models.pcfg_tools import (
    remove_epsilon_rules, remove_unary_rules)

from grammars.unmarked_reversal import UnmarkedReversalGrammar
from lower_bound_perplexity import compute_lower_bound_perplexity
from lang_algorithm.parsing import Parser
import random


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


#for str_len in []
grammar = UnmarkedReversalGrammar(2,20)
remove_epsilon_rules(grammar)
remove_unary_rules(grammar)


sampler = LengthSampler(grammar)
generator = random.Random()

X = np.asarray([list(sampler.sample(20, generator))
        for i in range(10000)])  


parser = Parser(grammar)
low_perp = compute_lower_bound_perplexity(sampler, parser, 1, X)   
    

#X = polyndrome(16)


X_ = np.zeros((X.shape[0], X.shape[1], 2))
X_[X == 1,1] = 1.
X_[X == 0,0] = 1. 
    
X = X_

y = X[:,1:]
X = X[:,:-1] 

print("Dataset: ", X.shape[0],X.shape[1])

x_train, x_validate, Y_train, Y_validate = train_test_split(
    X, y, test_size=0.5, shuffle=True
)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def ce_score(logits, labels):
    #labels_ = np.zeros((labels.shape[0], labels.shape[1], 2))
    #labels_[labels == 1,1] = 1.
    #labels_[labels == 0,0] = 1.                    
    ce = log_loss(labels.reshape(-1,2),logits.reshape(-1,2),normalize=False)
    return  np.exp(ce / (labels.shape[1] * labels.shape[0]))  

def make_model(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    output_layer = tf.keras.layers.SimpleRNN(20, return_sequences=True)(input_layer)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(output_layer)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

epochs = 200
batch_size = 10

model = make_model(input_shape=x_train.shape[1:])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

print("Simple RNN")
history = model.fit(
    x_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)

Y_v = model.predict(x_validate)


#print(
#    f"Boosted Cascade Classification report:\n"
#    f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")

print("Cross-entropy diff: ", np.log(ce_score(Y_v, Y_validate)) - np.log(low_perp))

def make_model2(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    output_layer = tf.keras.layers.LSTM(20, return_sequences=True)(input_layer)
    output_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2))(output_layer)    

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

epochs = 200
batch_size = 10

model = make_model2(input_shape=x_train.shape[1:])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
print("LSTM")
history = model.fit(
    x_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)


Y_v = model.predict(x_validate)


#print(
#    f"Boosted Cascade Classification report:\n"
#    f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")

print("Cross-entropy diff: ", np.log(ce_score(Y_v, Y_validate)) - np.log(low_perp))

# print("Boosted cascade")
# model = CascadeSequentialClassifier(C=1.0, n_layers=5, verbose=2, n_estimators = 4, max_depth=5,max_features='sqrt')#, n_iter_no_change = 1, validation_fraction = 0.1)
# 
# model.fit(x_train, Y_train)#, monitor = monitor)
# 
# Y_v = model.predict(x_validate)
# 
# 
# print(
#     f"Boosted Cascade Classification report:\n"
#     f"{metrics.classification_report(Y_validate.flatten(), Y_v.flatten())}\n")
# 
# print("Cross-entropy diff: ", np.log(ce_score(Y_v, Y_validate)) - np.log(low_perp))
