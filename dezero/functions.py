import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils


# =============================================================================
# Basic functions: sin / cos / tanh / exp
# =============================================================================
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


# =============================================================================
# sum / sum_to / broadcast_to  / matmul / linear 
# =============================================================================

class Sum(Function):

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class SumTo(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class Matmul(Function):

    def forward(self, X0, X1):
        Y = X0 @ X1
        return Y
    
    def backward(self, gy):
        X0, X1 = self.inputs
        gX0 = gy @ X1.T
        gX1 = X0.T @ gy
        return gX0, gX1

def matmul(X0, X1):
    return Matmul()(X0, X1)


class Linear(Function):
    def forward(self, x, W, b):
        y = x @ W
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = gy @ W.T
        gW = x.T @ gy
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None # tのデータを消去
    return y


# =============================================================================
# activation function: sigmoid 
# =============================================================================

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Sigmoid(Function):

    def forward(self, x):
        # y = 1 / (1+ exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)


# =============================================================================
# loss function: mean_squared_error
# =============================================================================

def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    y = sum(diff ** 2) / len(diff)
    return y

class MeanSquaredError(Function):

    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    # y = (x0 - x1)^2 / len(diff)
    # dy/dx0 =   2 (x0 - x1) / len(diff)
    # dy/dx1 = - 2 (x0 - x1) / len(diff)
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)