import ujson
import numpy as np

class Sender():

    def __init__(self):
        with open('../data_preprocess/goal.json') as f:
            self.data = ujson.load(f)
        self.start = 0
        self.end = 0

    def restart(self):
        self.start = 0

    def send(self,n):
        self.end = self.start+n
        if self.end > len(self.data):
            raise IndexError('epoch end')
        ret_data = np.array(self.data[self.start:self.end])
        self.start = self.end
        return list(ret_data[:,0]),list(ret_data[:,1]),list(ret_data[:,2])

if __name__ == '__main__':
    sender = Sender()
    for i in range(20):
        print(sender.send(1))