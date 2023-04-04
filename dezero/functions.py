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