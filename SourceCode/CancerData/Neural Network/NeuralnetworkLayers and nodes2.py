from NN_class import *


n_layers = [1,2,3,4,5]
#64,128,256,512,1024,2048
n_nodes = [1,8,16,32]
#n_nodes2 = [8,16,32,64,128,256,512,1024,1024,2048,1024,512]
eta_vals = np.logspace(-5, 1, 7)
lmb_vals = np.logspace(-5, 1, 7)

np.random.seed(1)
testAcc =np.zeros((len(n_nodes), len(lmb_vals)))
trainAcc=np.zeros((len(n_nodes), len(lmb_vals)))
for i, l in enumerate(n_nodes):
    size = np.empty(l)
    print("for layer ",l)
    size = [l]

    for j, k in enumerate(lmb_vals):

        LogReg = Network(XTrain, yTrain, num_iter=500, eta = 0.00001, lmbd = k, sizes = size)
        LogReg.backpropagation()
        LogReg.fitgradient()
        trainAcc[i][j]= LogReg.predict(XTrain,yTrain)
        testAcc[i][j] = LogReg.predict(XTest,yTest)
sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(trainAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("Hidden nodes")
ax.set_xlabel("Lambda")
ax.set_yticklabels(eta_vals)
ax.set_xticklabels(lmb_vals)
plt.show()

#values = ax.get_yticks()
#ax.set_yticklabels(["{0:.0%}".format(y/100) for y in values], fontdict={'fontweight': 'bold'})
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(testAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_yticklabels(eta_vals)
ax.set_xticklabels(lmb_vals)
ax.set_ylabel("eta")
ax.set_xlabel("Lambda")
plt.show()
