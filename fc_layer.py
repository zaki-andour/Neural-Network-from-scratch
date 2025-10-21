import numpy as np

from Actvation_Layer import Actvation_Layer


class Fc_layer:
    def _init_(self, _in, _out, activation_type):
        self.activation = Actvation_Layer(activation_type)
        self.weight = np.random.normal(0, 1, (_in + 1,_out))
        self.d_weight = None
        self._in = None

    def feedforward(self, X):
        if X.shape[1] != self.weight.shape[0] - 1: # FIRST X ONES
            raise Exception("Sorry, features don't correspond on the weights dimensions")
        else:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
            self._in = X[:]
            Z = np.dot(X, self.weight)
            return self.activation.feedforward(Z)
    def backprop(self,Y):
        dA = self.activation.backprop(Y) #result backpro layer
        Jacob =  self.weight.T
        print(self.weight.shape,self._in.shape,dA.shape)
        self.d_weight = np.dot(self._in.T,dA)
        return np.dot(dA,Jacob)[:,:1]

    def optimize(self, lr):
        self.weight=self.weight-lr*self.d_weight