import numpy as np


class Activation:
    def __init__(self, activation_name):
        self.name = activation_name
        self.feed_result = None
        self.feed_entry = None

    def feed_forward(self, x):
        self.feed_entry = x
        if self.name == 'relu':
            y = np.maximum(0, x)
            self.feed_result = y
            return y
        elif self.name == 'sigmoid':
            y = 1 / (1 + np.exp(-x))
            self.feed_result = y
            return y
        elif self.name == 'tanh':
            y = np.tanh(x)
            self.feed_result = y
            return y
        else:
            raise Exception("Sorry, No activation defined")

    def back_prop(self,y):
        if self.name == 'sigmoid':
            return y*self.feed_result*(1-self.feed_result)
        elif self.name == 'relu':
            return y*(self.feed_entry > 0).astype(int)
        elif self.name == 'tanh':
            return y*(1-self.feed_result**2)

