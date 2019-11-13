import pandas as pd
import numpy as np
import seaborn as sns
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
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

np.random.seed()
trainingShare = 0.9
seed  = 1

XTrain, XTest, yTrain, yTest=train_test_split(cancer.data,cancer.target, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)
yTrain = np.ravel(yTrain).reshape(-1,1)
yTest =  np.ravel(yTest).reshape(-1,1)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)
#PCA dimensionality reduction
'''
scaler = StandardScaler()
pca = PCA(n_components=2)
XTrain= pca.fit(XTrain).transform(XTrain)
XTest = pca.fit(XTest).transform(XTest)'''

class Network:
    def __init__(self,Xdata,Ydata,act_func = 'ELU',type_net = 'regression', sizes = [64,64], eta=0.001, lmbd = 0.01, num_iter= 100, n_epochs = 200, batch_size = 20):
        self.eta = eta #Learning rate
        self.num_iter = num_iter #Number of iterations
        #Parameters needed for stochastoc gradient
        self.n_epochs = n_epochs
        row, col = XTrain.shape
        tot_num_samples = row
        self.batch_size = batch_size
        self.batches = int(tot_num_samples/batch_size)
        #Activation function
        self.act_func = act_func
        #Type of network
        self.type_net = type_net
        #Data
        self.Xdata =Xdata
        self.Ydata =Ydata
        self.sizes = sizes #Number of layers
        self.lmbd = lmbd  #Regularization parameter
        #Definition of some parameters first for weights and bias
        self.column = [col]
        self.column1 =[int(i) for i in self.column]
        sizes = [int(i) for i in self.sizes]
        self.list0 = np.append(self.column1, self.sizes)
        self.list = np.append(self.list0,1)
        self.list2 = self.list[:-1]
        self.list3 = self.list[1:]
        #Make a loop for weights and biases
        self.listWeights = []
        self.biasesList = []
        for j, columns in enumerate(self.list2):
            #for rows in list3:
            rows = self.list3[j]
            weights = np.zeros((len(self.list2),len(self.list3)))
            weights = np.random.randn(columns, rows)
            self.listWeights.append(weights)
            bias = np.zeros((len(self.list3)))
            bias =  np.random.randn(rows)
            self.biasesList.append(bias)
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    #Derivative of sigmoid
    def sigmoid_prime(self,z):
        #Derivative of sigmoid function
        return self.sigmoid(z)*(1-self.sigmoid(z))
    #Definition of tanh function
    def tanh(self,z):
        return np.tanh(z)
    #Derivative of tanh function
    def derivtanh(self,z):
        return (1-(np.tanh(z)**2))
    #Definition of RELU
    def RELU(self, z):
        if (z.all() < 0):
            return 0
        else:
            return z
    #Derivative of RELU function
    def derivRELU(self, z):
        if (z.all() < 0):
            return 0
        else:
            return 1
    #Definition of ELU
    def ELU(self,z):
        if (z.all()<0):
            return alpha*(np.exp(z)-1)
        else:
            return z
    #Derivative of ELU
    def derivELU(self,z):
        if(z.all()<0):
            return (np.exp(z))
        else:
            return 1
    ##Return partial derivative with respect to activation for classification
    def cost_derivative(self, output_activations, Ydata):
        #Return partial derivative with respect to activation
            if (self.type_net) == 'Regression':
                return (output_activations - Ydata)
            else:
                return (output_activations - Ydata)/(output_activations*(1-output_activations))
    #Feed forward
    def feed_forward_train(self,):
        #Define my activation function
        activation = self.Xdata
        zs = []
        ac = [self.Xdata]
        for j, i in enumerate(self.listWeights):
            hidden_bias = self.biasesList[j]
            z = activation@i + hidden_bias
            #z_o = np.matmul(a_h, output_weights) + output_bias
            #z = np.matmul(activation.T, i) + hidden_bias
            zs.append(z) #
            #Specify which activation to use
            if (self.act_func) == 'tanh':
                activation = self.tanh(z)
            elif (self.act_func) == 'RELU':
                activation = self.RELU(z)
            elif (self.act_func) == 'ELU':
                activation = self.ELU(z)
            else:
                activation = self.sigmoid(z)
             # Creat list for weighted sum of inputs.
            ac.append(activation) # Create list for my activations
        return(ac, activation, zs)
    #Back propagation for gradient descent
    def backpropagation(self,):
        #I creat gradient for layer, by layer.
        nabla_w = [] # Creat an empty list for gradient in weight
        nabla_b = [] # Creat an empty list for gradient in bias
        for j, i in enumerate(self.listWeights):
            hidden_bias = self.biasesList[j]
            b = np.zeros(hidden_bias.shape)
            w = np.zeros(i.shape)
            nabla_b.append(nabla_b)
            nabla_w.append(nabla_w)
        #insert values from my feed_forward
        ac, output, zs = self.feed_forward_train()
        #error in output layer
            #Specify which activation to use

        if (self.act_func) == 'tanh':
            deriv = self.derivtanh(zs[-1])
        elif (self.act_func) == 'RELU':
            deriv = self.derivRELU(zs[-1])
        elif (self.act_func) == 'ELU':
            deriv = self.derivELU(zs[-1])
        else:
            deriv = self.sigmoid_prime(zs[-1])

        delta = self.cost_derivative(ac[-1],self.Ydata) * deriv
        #delta = (ac[-1] - self.Ydata)
        error = delta
        # Gradient bias for output layer
        nabla_b[-1] = np.sum(delta)
        #Gradient weight for weights
        nabla_w[-1] = np.dot(ac[-2].T,delta)
        #calculate gradient for hidden layers
        deltas = [error]
        for l in range(2, len(ac)):
            z = zs[-l]
            if (self.act_func) == 'tanh':
                deriv = self.derivtanh(z)
            elif (self.act_func) == 'RELU':
                deriv = self.derivRELU(z)
            elif (self.act_func) == 'ELU':
                deriv = self.derivELU(z)
            else:
                deriv = self.sigmoid_prime(z)
            sp = deriv
            delta = np.dot(delta,self.listWeights[-l+1].T)*sp
            nabla_b[-l] = np.sum(delta)
            nabla_w[-l] = np.dot(ac[-l-1].T,delta)
            deltas.append(delta)
            #deltas.append(delta)
        return (nabla_b, nabla_w) # I shall put values in stochastic gradient
    #Implement a backpropagation for stochastic gradient descent
    def backpropagation_stochastic(self,):
        #I creat gradient for layer, by layer.
        nabla_w = [] # Creat an empty list for gradient in weight
        nabla_b = [] # Creat an empty list for gradient in bias
        for j, i in enumerate(self.listWeights):
            hidden_bias = self.biasesList[j]
            b = np.zeros(hidden_bias.shape)
            w = np.zeros(i.shape)
            nabla_b.append(nabla_b)
            nabla_w.append(nabla_w)
        #insert values from my feed_forward
        ac, output, zs = self.feed_forward_output(self.Xdata1)
        #error in output layer
        if (self.act_func) == 'tanh':
            deriv = self.derivtanh(zs[-1])
        elif (self.act_func) == 'RELU':
            deriv = self.derivRELU(zs[-1])
        elif (self.act_func) == 'ELU':
            deriv = self.derivELU(zs[-1])
        else:
            deriv = self.sigmoid_prime(zs[-1])

        delta = self.cost_derivative(ac[-1],self.Ydata1) * deriv
        # Gradient bias for output layer
        nabla_b[-1] = np.sum(delta)
        #Gradient weight for weights
        nabla_w[-1] = np.dot(ac[-2].T,delta)
        #calculate gradient for hidden layers
        for l in range(2, len(ac)):
            z = zs[-l]
            if (self.act_func) == 'tanh':
                deriv = self.derivtanh(z)
            elif (self.act_func) == 'RELU':
                deriv = self.derivRELU(z)
            elif (self.act_func) == 'ELU':
                deriv = self.derivELU(z)
            else:
                deriv = self.sigmoid_prime(z)
            sp = deriv
            delta = np.dot(delta,self.listWeights[-l+1].T)*sp
            nabla_b[-l] = np.sum(delta)
            nabla_w[-l] = np.dot(ac[-l-1].T,delta)
            #deltas.append(delta)
        return (nabla_b, nabla_w) # I shall put values in stochastic gradient
    #Fitting with stochastic gradient descent
    def fitstoc(self,):
    	row, col = self.Xdata.shape
    	data_indices = np.arange(col) # Define my features
    	for epoch in range(self.n_epochs):
            for i in range(self.batches):
            	chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)
            	self.Xdata1 = self.Xdata[chosen_datapoints]
            	self.Ydata1 = self.Ydata[chosen_datapoints]
            	nabla_b, nabla_w = self.backpropagation_stochastic()
            	for m, l in enumerate(self.biasesList):
            		db = nabla_b[m]
            		l -=self.eta*db
            	dw = []
            	for j, k in enumerate(self.listWeights):
            		dwh = nabla_w[j]
            		dwh +=self.lmbd*k
            		dw.append(dwh) # Creat a list for regularized weights
            		k -=self.eta*dwh
    def fitgradient(self,):
        #Define parameters
        for i in range(self.num_iter):
            nabla_b, nabla_w = self.backpropagation()
            for m, l in enumerate(self.biasesList):
                db = nabla_b[m]
                l -= self.eta*db
            dw = []
            for j, k in enumerate(self.listWeights):
                dwh = nabla_w[j]
                dwh += self.lmbd*k
                dw.append(dwh) # Creat a list for regularized weights
                k -=self.eta*dwh

    def feed_forward_output(self, XTest):
        #Define my activation function
        activation = XTest
        zs = []
        ac = [XTest]
        for j, i in enumerate(self.listWeights):
            hidden_bias = self.biasesList[j]
            z = activation@i + hidden_bias
            zs.append(z) # Creat list for weighted sum of inputs.
            if (self.act_func) == 'tanh':
                activation = self.tanh(z)
            elif (self.act_func) == 'RELU':
                activation = self.RELU(z)
            elif (self.act_func) == 'ELU':
                activation = self.ELU(z)
            else:
                activation = self.sigmoid(z) #
            ac.append(activation) # Create list for my activations
        return(ac, activation, zs)

    def predict(self, XTRAIN,yTRAIN):
        #Call for activation function and call it ypredict
        ac, ypredict, zs = self.feed_forward_output(XTRAIN)
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
        C = np.ravel(C).reshape(-1,1)
        for i in range(0, row):
            if C[i] == yTRAIN[i]:
                a.append(1)
        return(len(a)/row)
    #Calculating R2 score
