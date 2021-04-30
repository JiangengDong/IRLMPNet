import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

from kinodyn_envs.visual.differential_drive import plot_differential_drive


def plot_boxplot():
    rlmpnettree_data = np.loadtxt("data/result/car1order/rlmpnettree_statistic_0.5.csv", delimiter=",")
    rlmpnet_data = np.loadtxt("data/result/car1order/rlmpnet_statistic_0.5.csv", delimiter=",")
    sst_data = np.loadtxt("data/result/car1order/sst_statistic_0.5.csv", delimiter=",")

    print(np.count_nonzero(np.isfinite(rlmpnet_data[:, 0])))
    print(np.count_nonzero(np.isfinite(rlmpnettree_data[:, 0])))
    print(np.count_nonzero(np.isfinite(sst_data[:, 0])))


    all_data = []
    all_data.append(pd.DataFrame({"Time": rlmpnet_data[:, 0], "Cost": rlmpnet_data[:, 1], "Algorithm": "RL-MPNet"}))
    all_data.append(pd.DataFrame({"Time": rlmpnettree_data[:, 0], "Cost": rlmpnettree_data[:, 1], "Algorithm": "RL-MPNetTree"}))
    all_data.append(pd.DataFrame({"Time": sst_data[:, 0], "Cost": sst_data[:, 1], "Algorithm": "SST"}))
    all_data = pd.concat(all_data)
    all_data = all_data.replace((np.inf, -np.inf), np.nan)
    all_data = all_data.dropna()

    f = plt.figure(figsize=(10, 5))
    axs = f.subplots(1, 2)

    ax = axs[0]
    sns.set(style="whitegrid", font="Times New Roman")
    sns.boxplot(x="Algorithm", y="Time", data=all_data,
                order=["SST", "RL-MPNet", "RL-MPNetTree"],
                whis=[0, 100], palette="BuGn", width=0.5, dodge=True, ax=ax, linewidth=1.0)
    ax.set_ylabel("time (second)")
    ax.set_xlabel("")
    sns.despine(left=True, bottom=True)

    ax = axs[1]
    sns.set(style="whitegrid", font="Times New Roman")
    sns.boxplot(x="Algorithm", y="Cost", data=all_data,
                order=["SST", "RL-MPNet", "RL-MPNetTree"],
                whis=[0, 100], palette="BuGn", width=0.5, dodge=True, ax=ax, linewidth=1.0)
    plt.ylabel("cost")
    plt.xlabel("")
    sns.despine(left=True, bottom=True)
    plt.savefig("data/result/statistic.pdf", format="pdf", bbox="tight")

def plot_gif():
    background = Image.new("RGB", (2300, 2300), color=(255, 255, 255))


if __name__ == "__main__":
    plot_boxplot()
