import matplotlib.pyplot as plt

def plot_curves(loss, train, act1, act2, act3, f_name):
    plt.clf()
    fig, axes = plt.subplots(nrows=2, ncols=1) 
    axes[0].plot(train, label = "Train Acc")
    axes[0].plot(val, label = "Train Loss")
    axes[0].xlabel("Epochs")
    axes[0].ylabel("Acc")
    axes[0].legend()

    axes[1].plot(act1, label = "Conv1 Spikes")
    axes[1].plot(act2, label = "Conv2 Spikes")
    axes[1].plot(act3, label = "FC Spikes")
    axes[1].xlabel("Epochs")
    axes[1].ylabel("# Spikes")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/"+ f_name + ".png")
    plt.close()
