import matplotlib.pyplot as plt
import numpy as np


def plot_distirbution(src):
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lengthes = [len(l.split()) for l in lines]
    print('Average ', np.sum(lengthes) / len(lengthes))
    plt.hist(lengthes, bins=range(0, max(lengthes)))
    plt.show()

# plot_distirbution('datasets/DailyDialog/train/Q_train.txt')
# plot_distirbution('datasets/DailyDialog/train/A_train.txt')
