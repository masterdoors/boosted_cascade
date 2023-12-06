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

from sklearn import preprocessing
from sklearn import datasets, metrics
import numpy as np
from sklearn.model_selection import train_test_split

#digits = datasets.load_digits()

#n_samples = len(digits.images)

#data = digits.images.reshape((n_samples, -1))

#Y =  np.asarray(digits.target).astype('int64')

#indexes = np.logical_or(Y == 1, Y == 0)
#data = data[indexes]
#Y = Y[indexes]

#print (np.unique(Y,return_counts=True))

#for i in range(len(Y)):
#    Y[i] = Y[i] + 1
    
#print(data.shape)    

#x = preprocessing.normalize(data, copy=False, axis = 0).reshape(-1,3,data.shape[1])
#Y = Y.reshape(-1,3)

#x_train, x_validate, Y_train, Y_validate = train_test_split(
#    x, Y, test_size=0.5, shuffle=True
#)


#X = np.asarray([[0,0,1,1,0],[1,0,0,0,0],[0,1,0,0,1],[1,0,1,1,0],[1,1,0,0,1],[0,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1]]).reshape(8,5,1)
#y = np.asarray([[0,1,1,0,0],[0,0,0,0,1],[1,0,0,1,0],[0,1,1,0,1],[1,0,0,1,1],[1,1,1,1,0],[0,0,0,0,0],[1,1,1,1,1]]).reshape(8,5)

def polyndrome(n):
    grids = []
    for k in range(0,n + 1):
        if k == 0:
            grid = np.zeros((1, 2*n), dtype="float")
        else:
            which = np.array(list(itertools.combinations(range(n), k)))    
            grid = np.zeros((len(which), 2*n), dtype="float")
        
            grid[np.arange(len(which))[None].T, which] = 1
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                grid[i,grid.shape[1] - j - 1] = grid[i,j]
        grids.append(grid)        
    return np.vstack(grids) 


simple_rnn_data = []
lstm_data = []
str_len = 20# in [20, 40, 60, 80, 100]:
#    for _ in range(5):
grammar = UnmarkedReversalGrammar(2,str_len)
remove_epsilon_rules(grammar)
remove_unary_rules(grammar)


sampler = LengthSampler(grammar)
generator = random.Random()

X = np.asarray([list(sampler.sample(str_len, generator))
        for i in range(1000)])
#X = polyndrome(7)  


parser = Parser(grammar)
low_perp = compute_lower_bound_perplexity(sampler, parser, 1, X)   
    




X_ = np.zeros((X.shape[0], X.shape[1], 2))
y = X[:,1:]

X_[X == 1,1] = 1.
X_[X == 0,0] = 1. 
    
X = X_


X = X[:,:-1] 

print("Dataset: ", X.shape[0],X.shape[1])

x_train, x_validate, Y_train, Y_validate = train_test_split(
    X, y, test_size=0.5, shuffle=True
)

#print(Y_train)
#print(Y_validate)

for y_ in Y_train:
    for y__ in Y_validate:
        if not np.logical_xor(y_, y__).sum():
            print(y_, y__)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def ce_score(logits, labels):
    labels_ = np.zeros((labels.shape[0], labels.shape[1], 2))
    labels_[labels == 1,1] = 1.
    labels_[labels == 0,0] = 1.                    
    ce = log_loss(labels_.reshape(-1,2),logits.reshape(-1,2),normalize=False)
    return  np.exp(ce / (labels.shape[1] * labels.shape[0]))  

def ce_score2(logits, labels):
    #labels_ = np.zeros((labels.shape[0], labels.shape[1], 2))            
    #labels_[labels == 1,1] = 1.
    #labels_[labels == 0,0] = 1.     
    #logits_ = sigmoid(logits)
    #logits_ = np.concatenate([logits_, 1. -logits_], axis=2)         
    ce =np.log(1 + np.exp(logits.flatten())) - labels.flatten() * logits.flatten()
    #print("test:", logits.shape, logits.max(), logits.mean(),logits.min(),labels.shape )    
    return  ce.mean()#np.exp(ce).mean()  

def make_model(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    initial_state = tf.keras.layers.Input((20,))
    output_layer = tf.keras.layers.SimpleRNN(20, return_sequences=True)(input_layer, initial_state=initial_state)
    output_layer2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation='softmax'))(output_layer)

    return tf.keras.models.Model(inputs=[input_layer] + [initial_state], outputs=output_layer2)

epochs = 100
batch_size = 10

model = make_model(input_shape=x_train.shape[1:])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

some_initial_state = np.zeros((x_train.shape[0], 20))
test_initial_state = np.zeros((x_validate.shape[0], 20))

print("Simple RNN")
history = model.fit(
    [x_train] + [some_initial_state],
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
)

Y_v = model.predict([x_validate] + [test_initial_state])


Y_v_labels = Y_v.argmax(axis=2)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v_labels.flatten())}\n")

simple_rnn_data.append((str_len, np.log(ce_score(Y_v, Y_validate)) - np.log(low_perp)))

Y_v = model.predict([x_train] + [some_initial_state], batch_size=batch_size)


Y_v_labels = Y_v.argmax(axis=2)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_train.flatten(), Y_v_labels.flatten())}\n")

simple_rnn_data.append((str_len, np.log(ce_score(Y_v, Y_train)) - np.log(low_perp)))
#print (simple_rnn_data)


print(simple_rnn_data)

def make_model2(input_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    dim = tf.zeros([batch_size,20])  
    output_layer = tf.keras.layers.LSTM(20, return_sequences=True)(input_layer, initial_state=[dim, dim])
    output_layer2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation='softmax'))(output_layer)    

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer2)

model = make_model2(input_shape=x_train.shape[1:])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
print("LSTM")
history = model.fit(
    x_train,
    Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
)

Y_v = model.predict(x_validate, batch_size=batch_size)


Y_v_labels = Y_v.argmax(axis=2)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v_labels.flatten())}\n")

lstm_data.append((str_len, np.log(ce_score(Y_v, Y_validate)) - np.log(low_perp)))
print (lstm_data)

Y_v = model.predict(x_train, batch_size=batch_size)


Y_v_labels = Y_v.argmax(axis=2)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_train.flatten(), Y_v_labels.flatten())}\n")

lstm_data.append((str_len, np.log(ce_score(Y_v, Y_train)) - np.log(low_perp)))
print (lstm_data)

print("Boosted cascade")
model = CascadeSequentialClassifier(C=100., n_layers=10, verbose=2, n_estimators = 4, max_depth=2,max_features='sqrt')#, n_iter_no_change = 1, validation_fraction = 0.1)


model.fit(x_train, Y_train)#, monitor = monitor)
 
Y_v = model.predict_proba(x_validate)
# 
# 

Y_v_labels = (Y_v >= 0.).astype(int)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_validate.flatten(), Y_v_labels.flatten())}\n")

print("Cross-entropy diff: ", ce_score2(Y_v, Y_validate) - np.log(low_perp))


Y_v = model.predict_proba(x_train)
# 
# 

Y_v_labels = (Y_v >= 0.).astype(int)

print(
    f"Boosted Cascade Classification report:\n"
    f"{metrics.classification_report(Y_train.flatten(), Y_v_labels.flatten())}\n")

print("Cross-entropy:", ce_score2(Y_v, Y_train))
print("Cross-entropy diff: ", ce_score2(Y_v, Y_train) - np.log(low_perp))

#print (simple_rnn_data)
#print (lstm_data)
