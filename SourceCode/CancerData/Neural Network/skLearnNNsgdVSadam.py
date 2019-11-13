import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

trainingShare = 0.8
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(cancer.data,cancer.target, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=0)
yTrain = np.ravel(yTrain)
yTest =  np.ravel(yTest)
# I removed .reshape(-1,1) from yTest and yTrain nebause it was causing me to get a warning

scaler.fit(XTrain)
XTrain = scaler.transform(XTrain)
XTest = scaler.transform(XTest)
'''
scaler = StandardScaler()
pca = PCA(n_components=2)
XTrain= pca.fit(XTrain).transform(XTrain)
XTest = pca.fit(XTest).transform(XTest)

#for a, c in enumerate(components):
pca = PCA(n_components=2)
XTrain= pca.fit(XTrain).transform(XTrain)
XTest= pca.fit(XTest).transform(XTest)'''

trainAcc1 = []
testAcc1 = []
trainAcc2 = []
testAcc2 = []
'''
# fit  gradient descent
LogReg = Network(XTrain, yTrain, num_iter=500, eta = 0.001, lmbd = 0.001,
sizes = [16,32],act_func = 'sigmoid')
LogReg.fitgradient()

# fit stochastic gradient descent
LogReg2 = Network(XTrain, yTrain, num_iter=500, eta = 0.001, lmbd = 0.001,
sizes = [16,32],act_func = 'sigmoid')'''

clf = MLPClassifier(solver='adam', alpha=0.001, max_iter=500,learning_rate_init=0.001,
                    hidden_layer_sizes=(5, 2), random_state=1, batch_size=50, activation = "logistic")

NN_numpy = clf.fit(XTrain, yTrain)

trainAcc1 =clf.score(XTrain, yTrain)
testAcc1 = clf.score(XTest, yTest)

clf2 = MLPClassifier(solver='sgd', alpha=0.001, max_iter=500,learning_rate_init=0.001,
                    hidden_layer_sizes=(5, 2), random_state=1, batch_size=50, activation = "logistic")

NN_numpy = clf2.fit(XTrain, yTrain)

trainAcc2 =clf2.score(XTrain, yTrain)
testAcc2 = clf2.score(XTest, yTest)

print("train GD",trainAcc1)
print("test GD",testAcc1)



print("train SGD",trainAcc2)
print("test GD",testAcc2)
