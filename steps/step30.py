if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

class Sin(Function):

    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

x = Variable(np.array(1.0))
y = sin(x)
x.name='x'
y.name='y'

plot_dot_graph(y, verbose=False, to_file='step30_sinx.png')