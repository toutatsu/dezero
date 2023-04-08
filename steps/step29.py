if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt


def f(x):
    y = x ** 4 - 2 * x ** 2 
    return y

# f(x)の2階微分
def gx2(x):
    return 12 * x ** 2 - 4

# 勾配降下法
x1 = Variable(np.array(2.0))
x1s=[]
# ニュートン法
x2 = Variable(np.array(2.0))
x2s=[]
lr = 0.01
iters = 200

# 勾配降下法とニュートン法の比較
for i in range(iters):

    print(i, x1, x2)
    x1s.append(x1.data.copy())
    x2s.append(x2.data.copy())

    y1 = f(x1)
    x1.cleargrad()
    y1.backward()

    y2 = f(x2)
    x2.cleargrad()
    y2.backward()

    x1.data -= lr * x1.grad
    x2.data -= x2.grad / gx2(x2.data)


# 可視化
# fig = plt.figure()
fig, axes = plt.subplots(ncols=2, figsize=(10,4), sharex=True, sharey=True)

axes[0].plot(np.linspace(-2.3, 2.3), f(np.linspace(-2.3, 2.3)))
axes[1].plot(np.linspace(-2.3, 2.3), f(np.linspace(-2.3, 2.3)))
axes[0].plot(x1s, f(np.array(x1s)), marker='.', markersize=10)
axes[1].plot(x2s, f(np.array(x2s)), marker='.', markersize=10)
axes[0].plot(1,f(1),marker='*', color='blue')
axes[1].plot(1,f(1),marker='*', color='blue')
plt.ylim(-2,10)
axes[0].set_title('gradient descent')
axes[1].set_title("Newton's method")
fig.savefig("step29_gradient_descent_and_Newton's_method.png")