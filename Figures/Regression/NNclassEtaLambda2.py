from NN_class import *


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n



eta_vals = [0.000001,0.00001,0.0001,0.001,0.1,1,5,10]
lmbd_vals = [0.000001,0.00001,0.0001,0.001,0.1,1,5,10]
#size list has different sizes for two hidden layes
size = [[1,1],[1,8],[8,1],[8,64],[16,32],[32,16],[64,1],[64,32],[128,32]]
R2 = [None] * len(size)
MSE = [None] * len(size)

#act_func = RELU or sigmoid or tanh or ELU
for s,siz in enumerate(size):
    MSE =np.zeros((len(eta_vals), len(lmbd_vals)))
    R2=np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            LogReg = Network(X, Z, num_iter=500, eta = eta, lmbd = lmbd,
            sizes = siz,act_func = 'RELU')
            LogReg.fitgradient()
            ac, z_tilde,zs = LogReg.feed_forward_output(X)
            R2[i][j]= R2(z,z_tilde)
            MSE[i][j] = MSE(z, z_tilde)
    R2[s] = R2
    MSE[s] = MSE
    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(R2, annot=True, ax=ax, cmap="viridis")
    ax.set_title("R2 score for nodes ({},{}) ".format(siz[0],siz[1]))
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis")
    ax.set_title("MSE for nodes ({},{}) ".format(siz[0],siz[1]))
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

'''
print("train acc",  trainAcc)

print("train acc Array",  trainAccArray[5])

for

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(trainAcc, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy for nodes = %i " %size[1])
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
'''
