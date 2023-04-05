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
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def clear_grad(self):
        self.grad = None

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
            gys = [output.grad for output in f.outputs] # 2. 関数の出力の勾配を取得
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

        self.generation = max([x.generation for x in inputs]) # 引数の中で最大の世代を自身に世代に設定
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
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)