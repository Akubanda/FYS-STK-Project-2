from NN_class import *

eta_vals = [0.000001,0.00001,0.0001,0.001,0.1,1,5,10]
lmbd_vals = [0.000001,0.00001,0.0001,0.001,0.1,1,5,10]
# = np.logspace(-5, 1, 7)
testAcc =np.zeros((len(eta_vals), len(lmbd_vals)))
trainAcc=np.zeros((len(eta_vals), len(lmbd_vals)))
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        LogReg = Network(XTrain, yTrain, num_iter=500, eta = eta, lmbd = lmbd, sizes = [128])
        LogReg.fitgradient()
        trainAcc[i][j]= LogReg.predict(XTrain,yTrain)
        testAcc[i][j] = LogReg.predict(XTest,yTest)

print("eta val", eta_vals)
print("lambda values",lmbd_vals)
sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(trainAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("$\eta$")
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
