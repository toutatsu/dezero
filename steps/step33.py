if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2 
    return y


# 勾配降下法
x = Variable(np.array(2.0))
xs = []
iters = 10

# ニュートン法
for i in range(iters):

    print(i, x)
    xs.append(x.data.copy())

    y = f(x)

    x.cleargrad()
    y.backward(create_graph=True)
    gx = x.grad
    
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data


# 可視化
import matplotlib.pyplot as plt
plt.plot(np.linspace(-2.3, 2.3), f(np.linspace(-2.3, 2.3)))
plt.plot(xs, f(np.array(xs)), marker='.', markersize=10)
plt.plot(1,f(1),marker='*', color='blue')
plt.ylim(-2,10)
plt.savefig("step33_Newton's_method.png")