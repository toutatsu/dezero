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
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):

    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


x = Variable(np.array(1.0))
y = sin(x)
y.backward(retain_grad=True, create_graph=True)
gx = x.grad
x.cleargrad()
gx.backward()

x.name = 'x'
gx.name='gx'
y.name = 'y'
y.grad.name='gy'

plot_dot_graph(gx, verbose=False, to_file='step31_computational_graph_of_gx.png')