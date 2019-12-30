import json
import matplotlib.pyplot as plt
import numpy as np


def show_data_distribution():
    with open('reason.txt') as f:
        reason = json.load(f)
    print(np.mean(reason))
    print(np.max(reason))
    print(np.min(reason))
    plt.plot(reason)
    plt.ylim(0, 2000)
    plt.show()


def right_rate(threshold):
    with open('reason.txt') as f:
        reason = json.load(f)
    a = np.mean(np.array(list(map(lambda x: int(x < threshold), reason))))
    return a


def show_threshold(step):
    x = []
    y = []
    for i in range(100):
        x.append(i * step)
        y.append(right_rate(i * step))
    plt.plot(x, y)
    # plt.xlim([0,0.004])
    # plt.ylim([0,1])
    plt.show()


if __name__ == '__main__':
    show_threshold(5)
