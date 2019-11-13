from FFNeuralNetClass import *

LogReg = Network(XTrain, yTrain, num_iter=500, eta = 0.0001, lmbd = 0.0001, sizes = [64,264])
LogReg.fitstoc()
prediction = LogReg.predict(XTest,yTest)
print(prediction)
