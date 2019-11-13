# Varying lambda and eta values to find optimal values
#import all from file with implementation of logReg class
from logRegClass import *

LogReg = LogisticRegression(XTrain, yTrain,lr=1, num_iter=1000, verbose = False)
#Optimize model with gradient descent
LogReg.fit()
#Optimize model with stochastic gradient descent
#Logeg.fitstoc()
test_predict = LogReg.predict(yTest, XTest)
print(test_predict)
