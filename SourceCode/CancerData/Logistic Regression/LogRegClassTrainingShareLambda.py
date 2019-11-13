import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import random, seed, randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cancer = load_breast_cancer()



#cancer = pd.DataFrame(cancer.data, columns = cancer.feature_names)
class LogisticRegression:
    def __init__(self, Xdata,Ydata,lr=0.0001, num_iter=100000, verbose = False,solver='saga'):
        self.lr = lr
        self.num_iter = num_iter
        self.Xdata =Xdata
        self.Ydata =Ydata

    #define my input values:-
    #Define my sigmoid function
    def sigmoid(self, z):
    # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-z))
    #Defining cost function
    def _cost(self, h):
        return (-self.Ydata * np.log(h) - (1 - self.Ydata) * np.log(-h))/(self.Ydata.shape[0])
    #Define gradient here.
    def _gradient(self, h):
        return np.dot(self.Xdata.T, (h-self.Ydata))/(self.Ydata.shape[0])
    #Perform gradient descent
    def fit(self,):
        row, col = self.Xdata.shape
        self.theta = np.zeros((col,1))
        #Creat an empy list for cost
        self.Cost = []
        for iter in range(self.num_iter):
            z = np.dot(self.Xdata, self.theta)
            h = self.sigmoid(z)
            gradients = self._gradient(h)
            cost = self._cost(h)
            self.Cost.append(cost)
            self.theta -= self.lr #* gradients

    def predict(self, YTest,XTest):
        ypredict = self.sigmoid(np.dot(XTest, self.theta))
        row, col = ypredict.shape
        #Creat an empty list for my prediction
        C = []
        for i in range(0, row):
            if ypredict[i] > 0.499999:
                C.append(1)
            else:
                C.append(0)
        #Empty list for testing
        a = []
        for i in range(0, row):
            if C[i] == YTest[i]:
                a.append(1)
        return(len(a)/row)


leguRate = np.logspace(-5, 1, 7)
tShare = [0.4, 0.5,0.75,0.8,0.9]
testAcc =np.zeros((len(tShare), len(leguRate)))
trainAcc=np.zeros((len(tShare), len(leguRate)))
for i, tS in enumerate(tShare):
    trainingShare = tS
    seed  = 1
    XTrain, XTest, yTrain, yTest=train_test_split(cancer.data,cancer.target, train_size=trainingShare, \
                                                  test_size = 1-trainingShare,
                                                 random_state=seed)
    yTrain = np.ravel(yTrain).reshape(-1,1)
    yTest =  np.ravel(yTest).reshape(-1,1)

    scaler.fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)
    pca = PCA(n_components=2)
    XTrain= pca.fit(XTrain).transform(XTrain)
    XTest = pca.fit(XTest).transform(XTest)
    #fit for PCA with 'comp' components and iterate for all learning rates
    for j, lr in enumerate(leguRate):
        LogReg = LogisticRegression(XTrain, yTrain,lr=lr,
                  num_iter=1000, verbose = False)

        LogReg.fit()
        testAcc[i][j] = LogReg.predict(yTest, XTest)
        #trainAcc[i][j] = LogReg.predict(XTrain,yTrain)

sns.set()
# train giving zeros
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(trainAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("Training share")
ax.set_xlabel("Regularization rate")
ax.set_yticklabels(tShare)
ax.set_xticklabels(leguRate)
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(testAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("Training share")
ax.set_xlabel("Regularization rate")
ax.set_yticklabels(tShare)
ax.set_xticklabels(leguRate)
plt.show()
