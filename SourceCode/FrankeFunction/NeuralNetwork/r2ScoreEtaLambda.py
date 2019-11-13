from FFNeuralNetClass import *

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

import seaborn as sns
act_fn = ['tanh','RELU','ELU','sigmoid']
eta_vals = [0.0000001, 0.00001,0.0001,0.001,0.01,0.1,0,1,5,10]
lmbd_vals =[0.0000001, 0.00001,0.0001,0.001,0.01,0.01,0,1,5,10]

for a, a_fn in enumerate(act_fn):
    mse =np.zeros((len(eta_vals), len(lmbd_vals)))
    R2score=np.zeros((len(eta_vals), len(lmbd_vals)))
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            LogReg = Network(X, z, act_func = act_fn,type_net = 'regression',
            num_iter=500, eta = eta, lmbd = lmbd, sizes = [32,8])
            LogReg.fitgradient()
            ac, z_tilde,zs = LogReg.feed_forward_output(X)
            R2score[i][j]= R2(z,z_tilde)
            mse[i][j] = MSE(z,z_tilde)


    sns.set()
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(R2score, annot=True, ax=ax, cmap="viridis")
    ax.set_title("R2-score with {} as activation function".format(a_fn))
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(R2score, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Mean Square Error for ")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    ax.set_yticklabels(eta_vals)
    ax.set_xticklabels(lmbd_vals)
    plt.show()
