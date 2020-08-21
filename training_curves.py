import matplotlib.pyplot as plt

def plot_curves(loss, train, act1, act2, act3, f_name):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1) 
    axes[0].plot(train, label = "Train Acc")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Acc")
    axes[0].legend()

    ax2 = axes[0].twinx() 
    ax2.plot(loss, label = "Train Loss")
    ax2.set_ylabel("Loss")

    axes[1].plot(act1, label = "Conv1 Spikes")
    axes[1].plot(act2, label = "Conv2 Spikes")
    axes[1].plot(act3, label = "FC Spikes")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("# Spikes")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/"+ f_name + ".png")
    plt.close()
