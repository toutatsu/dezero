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

    def clear_grad(self):
        self.grad = None

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs): # 可変長引数
        xs = [x.data for x in inputs] # データを取り出す
        ys = self.forward(*xs) # 実際の計算 可変長引数のアンパッキング
        if not isinstance(ys, tuple): # タプルではない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.inputs = inputs # 入力された変数を覚える
        self.outputs = outputs # 出力も覚える
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
    

class Add(Function):
    
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)



x = Variable(np.array(3.0))

# 1回目の計算_
y = add(x, x)
y.backward()
print(x.grad)

# 2回目の計算
x.clear_grad()
y = add(add(x, x), x)
y.backward()
print(x.grad)