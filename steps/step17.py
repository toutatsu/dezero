import weakref
import numpy as np
# import memory_profiler

class Variable:
    def __init__(self, data):

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self):

        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


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

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.inputs = inputs # 入力された変数を覚える
        self.outputs = [weakref.ref(output) for output in outputs] # 出力も覚える
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


for i in range(10):
    x = Variable(np.random.randn(10000)) # 巨大なデータ
    y = square(square(square(x))) # 複雑な計算をする