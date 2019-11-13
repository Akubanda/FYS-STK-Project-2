from NN_class import *


trainAcc1 = []
testAcc1 = []
trainAcc2 = []
testAcc2 = []
# fit  gradient descent
LogReg = Network(XTrain, yTrain, num_iter=500, eta = 0.001, lmbd = 0.001,
sizes = [16,32],act_func = 'sigmoid')
LogReg.fitgradient()
trainAcc1.append(LogReg.predict(XTrain,yTrain))
testAcc1.append(LogReg.predict(XTest,yTest))

# fit stochastic gradient descent
LogReg2 = Network(XTrain, yTrain, num_iter=500, eta = 0.001, lmbd = 0.001,
sizes = [16,32],act_func = 'sigmoid')
LogReg2.fitstoc()
trainAcc2.append(LogReg2.predict(XTrain,yTrain))
testAcc2.append(LogReg2.predict(XTest,yTest))

print("train GD",trainAcc1)
print("test GD",testAcc1)

print("train SGD",trainAcc2)
print("test GD",testAcc2)
