from fc_layer import fc_layer


class nn:
    def __init__(self):
        self.blocks = []

    def add_layer(self, _in, _out, activation_type):
        if self.blocks != [] and self.blocks[-1].weights.shape[0] != _in:
            raise Exception('You have a problem of compatibility dimension')
        else:
            self.blocks.append(fc_layer(_in, _out, activation_type))

    def feed_forward(self, x):
        for layer in self.blocks:
            x = layer.feed_forward(x)
        return x
    def backprop(self,Y):
        for layer in reversed(self.blocks):
            Y=layer.backprop(Y)
        

