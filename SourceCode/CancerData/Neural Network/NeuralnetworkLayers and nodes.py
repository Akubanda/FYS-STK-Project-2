from NN_class import *

n_layers = [1,2,3,4,5]
n_nodes = [1,8,16,32,64,128,256,512,1024,2048]
#n_nodes2 = [8,16,32,64,128,256,512,1024,1024,2048,1024,512]
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

np.random.seed(1)
testAcc =np.zeros((len(n_nodes), len(lmb_vals)))
trainAcc=np.zeros((len(n_nodes), len(lmb_vals)))
for i, l in enumerate(n_nodes):
    size = np.empty(l)
    print("for layer ",l)
    size = [l]

    for j, k in enumerate(lmb_vals):
        '''
        if (l==1):
            size = [n_nodes[j]]
        if (l==2):
            size = [n_nodes[j],n_nodes2[j]]
        if (l==3):
            size = [n_nodes[j],n_nodes2[j],n_nodes2[j]]
        if (l==4):
            size = [n_nodes[j],n_nodes2[j],n_nodes2[j],n_nodes2[j+1]]
        if (l==5):
            size = [n_nodes[j],n_nodes[j],n_nodes[j],n_nodes2[j+1],n_nodes2[j+2]]
        #print(size) '''

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
ax.set_yticklabels(n_nodes)
ax.set_xticklabels(lmbd_values)
plt.show()

#values = ax.get_yticks()
#ax.set_yticklabels(["{0:.0%}".format(y/100) for y in values], fontdict={'fontweight': 'bold'})
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(testAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_yticklabels(n_nodes)
ax.set_xticklabels(lmbd_values)
ax.set_ylabel("Hidden nodes")
ax.set_xlabel("Lambda")
plt.show()
