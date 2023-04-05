import numpy as np


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    def __init__(self, data):

        # np.ndarrayだけ扱う
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):

        # 逆伝播の最初の変数に勾配を設定
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # recursion
        # f = self.creator # 1. 関数を取得
        # if f is not None:
        #     x = f.input # 2. 関数の入力を取得
        #     x.grad = f.backward(self.grad) # 3. 関数のbackwardメソッドを呼ぶ
        #     x.backward() # 4. 自分より1つ前の変数のbackwardメソッドを呼ぶ (再帰)

        # list
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 1. 関数を取得
            x, y = f.input, f.output # 2. 関数の入出力を取得
            x.grad = f.backward(y.grad) # 3. 関数のbackwardメソッドを呼ぶ
            if x.creator is not None:
                funcs.append(x.creator) # 4. 1つ前の関数をリストに追加


# Variableインスタンスに変換
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# ndarrayインスタンスに変換
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



# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================

class Add(Function):
    
    def forward(self, x0, x1):
        y = x0 + x1
        return y

def add(x0, x1):
    return Add()(x0, x1)