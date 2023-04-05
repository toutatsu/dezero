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
    def __call__(self, inputs):
        xs = [x.data for x in inputs] # データを取り出す
        ys = self.forward(xs) # 実際の計算
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.inputs = inputs # 入力された変数を覚える
        self.outputs = outputs # 出力も覚える
        return outputs

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
    

class Add(Function):
    
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
    

xs = [Variable(np.array(2)), Variable(np.array(3))] # リストとして準備
f = Add()
ys = f(xs)
y = ys[0]
print(y.data)