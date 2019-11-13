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

cancer = load_breast_cancer()

np.random.seed(0)
trainingShare = 0.5 
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(cancer.data,cancer.target, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)
yTrain = np.ravel(yTrain).reshape(-1,1)
yTest =  np.ravel(yTest).reshape(-1,1)

# Input Scaling
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
#print('Xtrain', XTrain)
XTest = sc.transform(XTest)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from random import random, seed
from sklearn.utils import resample

np.random.seed(2018)
n = 40
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.randn(n,n)
z = FrankeFunction(x, y)

z = np.ravel(z)
z = z.reshape(-1, 1)
#Design matrix
def CreateDesignMatrix_X(x, y, n):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)      # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X
maxdegree = 5
X = CreateDesignMatrix_X(x, y, maxdegree)
print('for degree = ', maxdegree,',shape of matrix is:',X.shape)

#Split of our data
#Split of the data. Is the way I am splitting my data correct?
XTrain, XTest, yTrain, yTest = train_test_split(X, z, test_size=0.2)

class Network:
    def __init__(self,Xdata,Ydata,act_func = 'sigmoid',type_net = 'regression', sizes = [64,64], eta=0.001, lmbd = 0.01, num_iter= 500, n_epochs = 200, batch_size = 20):
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
            return (np.exp(z)-1)
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
        #Error in output layer
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
            #Choice of different activation
            if (self.act_func) == 'tanh':
                deriv = self.derivtanh(z)
            elif (self.act_func) == 'RELU':
                deriv = self.derivRELU(z)
            elif (self.act_func) == 'ELU':
                deriv = self.derivELU(z)
            else:
                deriv = self.sigmoid_prime(z)
            sp = deriv
            #Error in hidden layers.
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
    #Calculating R2 s


