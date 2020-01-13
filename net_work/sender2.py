import ujson
import numpy as np
import random
class Sender():

    def __init__(self):
        with open('../data_preprocess/goal2.json') as f:
            self.data = ujson.load(f)
        self.start = 0
        self.end = 0
        self.length = len(self.data)
        self.test = self.data[int(0.8*self.length):]
        self.data = self.data[:int(0.8*self.length)]


    def restart(self):
        self._random()
        self.start = 0

    def _random(self):
        self.data = random.sample(self.data,int(0.8*self.length))


    def send(self,n):
        self.end = self.start+n
        if self.end > len(self.data):
            print(len(self.data))
            raise Exception('epoch end')
        ret_data = np.array(self.data[self.start:self.end])
        self.start = self.end
        return ret_data[:,0].tolist(),ret_data[:,1].tolist(),ret_data[:,2].tolist()



if __name__ == '__main__':
    sender = Sender()
    for i in range(20):
        print(sender.send(1))