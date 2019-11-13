from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import all from file with implementation of logReg class
from logRegClass import *
numIter=[1,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]

testAcc =np.zeros(len(numIter))
trainAcc=np.zeros(len(numIter))
testAcc2 =np.zeros(len(numIter))
trainAcc2=np.zeros(len(numIter))
for i, e in enumerate(numIter):

    LogReg = LogisticRegression(XTrain, yTrain,lr=0.1, num_iter=e, verbose = False)
    print(yTrain.shape)
    #Optimize model with gradient descent
    LogReg.fit()
    testAcc[i] = LogReg.predict(yTest, XTest)
    trainAcc[i]= LogReg.predict(yTrain, XTrain)
    #Optimize model with stochastic gradient descent
    LogReg2 = LogisticRegression(XTrain, yTrain,lr=0.1, num_iter=e, verbose = False)
    LogReg2.fit()
    LogReg2.fitstoc()
    testAcc2[i] = LogReg2.predict(yTest, XTest)
    trainAcc2[i]= LogReg2.predict(yTrain, XTrain)

plt.plot(numIter, trainAcc, label='GD Train Accuracy')
plt.plot(numIter, testAcc, label='GD Test Accuracy')
plt.plot(numIter, trainAcc2, label='SGD Train Accuracy')
plt.plot(numIter, testAcc2, label='SGD Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
