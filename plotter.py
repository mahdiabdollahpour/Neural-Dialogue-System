import matplotlib.pyplot as plt
import numpy as np


def plot_distribution(src):
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lengthes = [len(l.split()) for l in lines]
    print('Average ', np.sum(lengthes) / len(lengthes))
    plt.hist(lengthes, bins=range(0, max(lengthes)))
    plt.show()


def plot_loss(src):
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        x = [int(line.split(',')[1]) for line in lines]
        y = [float(line.split(',')[0]) for line in lines]
        plt.plot(x, y,'-o')
        plt.show()
# plot_distribution('datasets/DailyDialog/train/Q_train.txt')
# plot_distribution('datasets/DailyDialog/train/A_train.txt')
# plot_loss('logs_v3logs.txt')