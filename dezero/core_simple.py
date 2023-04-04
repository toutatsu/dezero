import numpy as np


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


# ndarrayインスタンスに変換
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data # データを取り出す
        y = self.forward(x) # 実際の計算
        output = Variable(as_array(y))
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input # 入力された変数を覚える
        self.output = output # 出力も覚える
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()