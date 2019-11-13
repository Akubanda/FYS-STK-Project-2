import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from random import random, seed, randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

trainingShare = 0.7
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(cancer.data,cancer.target, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)
yTrain = np.ravel(yTrain).reshape(-1,1)
yTest =  np.ravel(yTest).reshape(-1,1)

print('Shape is', XTrain.shape)
print('Helo', XTest.shape)

yTrain = np.ravel(yTrain).reshape(-1,1)
yTest =  np.ravel(yTest).reshape(-1,1)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
#print('Xtrain', XTrain)
XTest = sc.transform(XTest)

#cancer = pd.DataFrame(cancer.data, columns = cancer.feature_names)
class LogisticRegression:
    def __init__(self, Xdata,Ydata,lr=0.0001, num_iter=1000, batch_size = 20, verbose = False):
        self.lr = lr
        self.num_iter = num_iter
        self.Xdata =Xdata
        self.Ydata =Ydata
        row, col = XTrain.shape
        tot_num_samples = row
        self.batch_size = batch_size
        self.batches = int(tot_num_samples/batch_size)

    #Define my sigmoid function
    def sigmoid(self, z):
    # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-z))
    #Defining cost function
    def _cost(self, h):
        return (-self.Ydata * np.log(h) - (1 - self.Ydata) * np.log(-h))/(self.Ydata.shape[0])
    #Define gradient here.
    def _gradient(self,h):
        return np.dot(self.Xdata.T, (h-self.Ydata))/(self.Ydata.shape[0])
    #Perform gradient descent.
    def fit(self,):
        row, col = self.Xdata.shape
        self.theta = np.zeros((col,1))#Create initial weights
        #Creat an empy list for cost
        self.Costgrad = []
        for iter in range(self.num_iter):
            z = np.dot(self.Xdata, self.theta) # Define weighted sum
            h = self.sigmoid(z) #Shoot it in sigmoid functio
            gradients = self._gradient(h)
            cost = self._cost(h)
            self.Costgrad.append(cost)
            self.theta -= self.lr * gradients
    #Here i implement stochastic gradient descent
    def fitstoc(self,):
        row, col = self.Xdata.shape
        self.theta = np.zeros((col,1))#Create initial weights
        data_indices = np.arange(col) #Define my features
        self.Costgradstoc = [] # Define empty loop for stochastic cost
        for epoch in range(self.num_iter):
            for i in range(self.batches):
                #Choose random data points without replacing
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)
                xi = XTrain[chosen_datapoints]
                yi = yTrain[chosen_datapoints]
                Ydata = yi
                z = np.dot(xi, self.theta) # Define weighted sum
                h = self.sigmoid(z) #Shoot it in sigmoid function
                #gradients = (np.transpose(xi)@(self.sigmoid(z)-yi))#/(yi.shape[0])
                gradients = ((np.transpose(xi)@(self.sigmoid(z) - yi)))/(yi.shape[0])
                #cost = (-yi * np.log(h) - (1 - yi) * np.log(-h))/(yi[0])
                #self.Costgradstoc.append(cost)
                self.theta -= self.lr * gradients

    def predict(self, YTest,XTest):
        ypredict = self.sigmoid(np.dot(XTest, self.theta))
        row, col = ypredict.shape
        #Creat an empty list for my prediction
        C = []
        for i in range(0, row):
            if ypredict[i] > 0.5:
                C.append(1)
            else:
                C.append(0)
        #Empty list for testing
        a = []
        for i in range(0, row):
            if C[i] == YTest[i]:
                a.append(1)
        return(len(a)/row)
