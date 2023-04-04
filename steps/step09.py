import numpy as np


class Variable:
    def __init__(self, data):

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    # y = x^2
    def forward(self, x):
        return x ** 2
    # dy/dx = 2x
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


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


x = Variable(np.array(1.0)) # OK
x = Variable(None) # OK
# x = Variable(1.0) # NG : エラーが発生!

x = Variable(np.array(0.5))

# 順伝播
y = square(exp(square(x)))

# 逆伝播
y.grad = np.array(1.0)
y.backward()
print(x.grad)