if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt


def rosenblock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001 # 学習率
iters = 10000 # 繰り返す回数

x0s=[]
x1s=[]

# 勾配降下法
for i in range(iters):

    print(x0, x1)
    x0s.append(x0.data.copy())
    x1s.append(x1.data.copy())

    y = rosenblock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

# 勾配降下の可視化
fig = plt.figure()

X, Y = np.meshgrid(np.arange(-2, 2, 0.01), np.arange(-1, 3, 0.01))
Z = rosenblock(X, Y)
cont = plt.contour(X, Y, Z, levels=[2 ** i for i in range(9)])

plt.plot(x0s,x1s,marker='.', color='orange', markersize=15)
plt.plot(1,1,marker='*', color='blue', markersize=15)

fig.savefig("step28_gradient_descent.png")