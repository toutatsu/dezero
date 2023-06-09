import weakref
import numpy as np
import contextlib
import dezero


# =============================================================================
# Config
# =============================================================================
class Config():
    enable_backprop = True
    train = True


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


def test_mode():
    return using_config('train', False)


# =============================================================================
# Variable / Function
# =============================================================================

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    __array_priprity__ = 200 # 演算子の優先度
    def __init__(self, data, name=None):

        # array_typesだけ扱う
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        return f'variable({str(self.data)})'.replace('\n', '\n' + ' ' * 9) 

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def unchain(self):
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):

        # 逆伝播の最初の変数に勾配を設定
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data)) # 高階微分を求める計算グラフを構築するためVariable

        # recursion
        # f = self.creator # 1. 関数を取得
        # if f is not None:
        #     x = f.input # 2. 関数の入力を取得
        #     x.grad = f.backward(self.grad) # 3. 関数のbackwardメソッドを呼ぶ
        #     x.backward() # 4. 自分より1つ前の変数のbackwardメソッドを呼ぶ (再帰)

        # 関数リスト
        funcs = []
        # funcsに同じ関数が追加されるのを防ぐ
        seen_set = set()

        # 関数fをfuncsに追加してgeneration順に並べる
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 1. 関数を取得

            # 1変数関数の逆伝播
            # x, y = f.input, f.output # 2. 関数の入出力を取得
            # x.grad = f.backward(y.grad) # 3. 関数のbackwardメソッドを呼ぶ

            # if x.creator is not None:
            #     funcs.append(x.creator) # 4. 1つ前の関数をリストに追加

            # 多変数関数の逆伝播
            gys = [output().grad for output in f.outputs] # 2. 関数の出力の勾配を取得

            with using_config('enable_backprop', create_graph): # create_graph=Trueで逆伝播時の計算に対する計算グラフも作成

                gxs = f.backward(*gys) # 3. 関数のbackwardメソッドを呼ぶ
                if not isinstance(gxs, tuple): # タプルではない場合の追加対応
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs): # 4. 関数の入力の勾配を設定
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None: # 5. 前の関数をリストに追加
                        add_func(x.creator)

            if not retain_grad: # 微分を伝えてきた変数の微分情報を削除
                for y in f.outputs:
                    y().grad = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes, (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)
    
    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)


class Parameter(Variable):
    pass


# Variableインスタンスに変換
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# ndarrayインスタンスに変換
def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs): # 可変長引数
        inputs = [as_variable(x) for x in inputs] # Variableインスタンスに変換
        xs = [x.data for x in inputs] # データを取り出す
        ys = self.forward(*xs) # 実際の計算 可変長引数のアンパッキング
        if not isinstance(ys, tuple): # タプルではない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop: # 逆伝播が有効の場合  
            self.generation = max([x.generation for x in inputs]) # 引数の中で最大の世代を自身に世代に設定
            for output in outputs:
                output.set_creator(self) # 出力変数に生みの親を覚えさせる
            self.inputs = inputs # 入力された変数を覚える
            self.outputs = [weakref.ref(output) for output in outputs] # 出力も弱参照で覚える

        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()



# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================

class Add(Function):
    
    def forward(self, x0, x1):
        # self.x0_shape, self.x1_shape = x0.shape, x1.shape # self.inputs[n].shapeで代用
        y = x0 + x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if x0.shape != x1.shape: # broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data)) # ndarrayインスタンスに変換
    return Add()(x0, x1)


class Mul(Function):

    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape: # broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):

    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)


class Sub(Function):

    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy
        gx1 = -gy
        if x0.shape != x1.shape: # broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0) # x1, x0を入れ替え


class Div(Function):

    def forward(self, x0, x1):
        y = x0 / x1
        return y 

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * 1 / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape: # broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0) # x1, x0を入れ替え


class Pow(Function):

    def __init__(self, c):
        self.c = c # 冪指数c 定数扱い

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = gy * c * x ** (c-1)
        return gx

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__add__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__matmul__ = dezero.matmul
    Variable.__getitem__ = dezero.functions.get_item