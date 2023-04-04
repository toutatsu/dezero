import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data # データを取り出す
        y = self.forward(x) # 実際の計算
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()