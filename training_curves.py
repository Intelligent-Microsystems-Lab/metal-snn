import matplotlib.pyplot as plt

def plot_curves(acc_hist, aux_hist, clal_hist, auxl_hist, act1_hist, act2_hist, act3_hist, model_uuid):
    plt.clf()
    fig, axes = plt.subplots(nrows=3, ncols=1) 
    axes[0].plot(clal_hist, label = "Classification Loss", color = 'tab:blue')
    axes[0].plot(auxl_hist, label = "Rotation Loss", color = 'tab:orange')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(acc_hist, label = "Classification Acc", color = 'tab:blue')
    axes[1].plot(aux_hist, label = "Rotation Acc", color = 'tab:orange')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Acc")
    axes[1].legend()

    axes[2].plot(act1_hist, label = "Conv1 Spikes")
    axes[2].plot(act2_hist, label = "Conv2 Spikes")
    axes[2].plot(act3_hist, label = "FC Spikes")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("# Spikes")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("figures/"+ model_uuid + ".png")
    plt.close()
