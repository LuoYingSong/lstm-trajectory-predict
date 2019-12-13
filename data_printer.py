import matplotlib.pyplot as plt
from data_sender import sender
import json
import numpy as np

data, _ = sender.next_batch(1)
data = data[:, :].reshape(-1, 2)
x = np.array([])
y = np.array([])
# print(data[:,1])

while len(data):
    x = np.append(x, data[:, 0])
    y = np.append(y, data[:, 1])
    # plt.scatter(data[:, 0], data[:, 1])
    data, _ = sender.next_batch(10)
    data = data[:, :].reshape(-1, 2)

plt.scatter(x, y, s=1,alpha=0.005)
# plt.colorbar()
plt.show()
