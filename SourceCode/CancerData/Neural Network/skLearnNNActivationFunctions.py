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
''' I noticed that when hidden_layer_vals = [[10,50],[2,5],[10,10]] it was only
the identity fn that performed well'''

hidden_layer_vals = [[8],[64],[8,32],[32,64],[64,8],[8,8,8],[32,32,32],[8,32,64],[64,32,8]]
act_fn=('identity', 'logistic', 'tanh', 'relu')

testAcc =np.zeros((len(act_fn), len(hidden_layer_vals)))
trainAcc=np.zeros((len(act_fn), len(hidden_layer_vals)))

#for a, c in enumerate(components):
pca = PCA(n_components=2)
XTrain= pca.fit(XTrain).transform(XTrain)
XTest= pca.fit(XTest).transform(XTest)

for i, act in enumerate(act_fn):
    for j, h_size in enumerate( hidden_layer_vals):
        clf = MLPClassifier(solver='sgd', alpha=0.0001, max_iter=5,
        learning_rate_init=0.001,activation=act,hidden_layer_sizes=h_size,
        random_state=1, batch_size=50)

        fit = clf.fit(XTrain, yTrain)

        trainAccuracy = clf.score(XTrain, yTrain)
        testAccuracy = clf.score(XTest, yTest)
        #print("train accuracy is ", trainAccuracy)
        #print("test accuracy is ", testAccuracy)
        #trainAcc.append(trainAccuracy)
        #testAcc.append(testAccuracy)
        testAcc[i][j]= testAccuracy
        trainAcc[i][j]= trainAccuracy


print("train accuracy", trainAcc)
print("test accuracy",testAcc)
sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(trainAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("Activation Function")
ax.set_xlabel("hidden values")
ax.set_yticklabels(act_fn)
ax.set_xticklabels(hidden_layer_vals)
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(testAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("Activation Function")
ax.set_xlabel("hidden values")
ax.set_yticklabels(act_fn)
ax.set_xticklabels(hidden_layer_vals)
plt.show()
