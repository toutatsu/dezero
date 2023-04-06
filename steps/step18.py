import weakref
import numpy as np
# import memory_profiler
import contextlib


class Config():
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


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

    def backward(self, retain_grad=False):

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

                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None


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


class Add(Function):
    
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


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


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad) # None None
print(x0.grad, x1.grad) # 2.0 1.0


with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)