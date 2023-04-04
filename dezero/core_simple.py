import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 1. 関数を取得
        if f is not None:
            x = f.input # 2. 関数の入力を取得
            x.grad = f.backward(self.grad) # 3. 関数のbackwardメソッドを呼ぶ
            x.backward() # 4. 自分より1つ前の変数のbackwardメソッドを呼ぶ (再帰)


class Function:
    def __call__(self, input):
        x = input.data # データを取り出す
        y = self.forward(x) # 実際の計算
        output = Variable(y)
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input # 入力された変数を覚える
        self.output = output # 出力も覚える
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()