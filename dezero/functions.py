import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils


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


class Tanh(Function):

    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)


class Exp(Function):
    # y = e^x
    def forward(self, x):
        y = np.exp(x)
        return y

    # dy/dx = e^x
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)


# =============================================================================
# Tensor operations: reshape / transpose 
# =============================================================================

class Reshape(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):

    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes) # numpyのtranspose
        return y

    def backward(self, gy):
        if self.axes == None:
            return transpose(gy)

        # axes_len = len(self.axes)
        # inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        inv_axes = tuple(np.argsort(self.axes)) # TODO: test axes_lenで割らない場合
        return transpose(gy, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)