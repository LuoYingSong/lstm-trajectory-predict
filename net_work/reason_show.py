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
    plt.show()

def trans(ptr1, ptr2):
    print(-ptr1[0]+ptr2[0],-ptr1[1]+ptr2[1])

if __name__ == '__main__':
    print(right_rate(100))
    show_threshold(2.0)
    # trans([119.11453300000001,26.121075],[119.116698,26.11919])