from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import all from file with implementation of logReg class
from logRegClass import *
lrArray = [1e-5,1e-4,1e-3,1e-2,1e-1,1]
testAccGD =np.zeros(len(lrArray))
trainAccGD=np.zeros(len(lrArray))
testAccSGD =np.zeros(len(lrArray))
trainAccSGD=np.zeros(len(lrArray))
for i, e in enumerate(lrArray):

    LogReg = LogisticRegression(XTrain, yTrain,lr=e, num_iter=1000, verbose = False)
    print(yTrain.shape)
    #Optimize model with gradient descent
    LogReg.fit()
    testAccGD[i] = LogReg.predict(yTest, XTest)
    trainAccGD[i]= LogReg.predict(yTrain, XTrain)
    #Optimize model with stochastic gradient descent
    LogReg2 = LogisticRegression(XTrain, yTrain,lr=e, num_iter=1000, verbose = False)
    LogReg2.fit()
    LogReg2.fitstoc()
    testAccSGD[i] = LogReg2.predict(yTest, XTest)
    trainAccSGD[i]= LogReg2.predict(yTrain, XTrain)
print("----------------for varying eta values-------------")
print(lrArray)
print("train Accuracy GD:",trainAccGD)
print("test Accuracy GD:",testAccGD)
print("train Accuracy SGD:",trainAccSGD)
print("test Accuracy SGD:",testAccSGD)

'''
plt.plot(numIter, trainAcc, label='GD Train Accuracy')
plt.plot(numIter, testAcc, label='GD Test Accuracy')
plt.plot(numIter, trainAcc2, label='SGD Train Accuracy')
plt.plot(numIter, testAcc2, label='SGD Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show() '''
