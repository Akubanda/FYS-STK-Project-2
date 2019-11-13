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
#components =[10,5,4,2]
#learning_rate = [50,10,1,0.1,0.5,0.001,0.0001,0.00001]
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
testAcc =np.zeros((len(eta_vals), len(lmbd_vals)))
trainAcc=np.zeros((len(eta_vals), len(lmbd_vals)))

#for a, c in enumerate(components):
pca = PCA(n_components=2)
XTrain= pca.fit(XTrain).transform(XTrain)
XTest= pca.fit(XTest).transform(XTest)
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate( lmbd_vals):
        clf = MLPClassifier(solver='sgd', alpha=lmbd, max_iter=5,learning_rate_init=eta,
                            hidden_layer_sizes=(5, 2), random_state=1, batch_size=50)

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
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_yticklabels(eta_vals)
ax.set_xticklabels(lmbd_vals)
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(testAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
ax.set_yticklabels(eta_vals)
ax.set_xticklabels(lmbd_vals)
plt.show()
'''
for a,b in enumerate()
plt.plot(components, trainAcc, label='linear')
plt.legend()
plt.show()
''''''
def models(XTrain, yTrain,XTest,yTest):
    #logistic regression
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression()
    logreg.fit(XTrain,yTrain)
    #add more models, compare their accuracy
    logregScore = logreg.score(XTrain,yTrain)
    return logregScore

models(XTrain, yTrain,XTest,yTest)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(yTest,models.predict(XTest))
print(cm)
'''
