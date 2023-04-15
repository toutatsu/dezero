if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

# 線形回帰
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = x @ W + b
    return y

# 平均二乗誤差
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

# 計算グラフ
x.name = 'x'
y.name = 'y'
W.name = 'W'
b.name = 'b'

plot_dot_graph(
    mean_squared_error(y, predict(x)),
    verbose=False,
    to_file=f"step42_MSE_simple.png"
)
plot_dot_graph(
    F.mean_squared_error(y, predict(x)),
    verbose=False,
    to_file=f"step42_MSE.png"
)

# 学習
lr = 0.1
iters = 100
y_pred_history = []
W_history = []
b_history = []
loss_history = []

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    # loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

    y_pred_history.append(y_pred.data.copy())
    W_history.append(W.data[0][0].copy())
    b_history.append(b.data[0].copy())
    loss_history.append(loss.data.copy())


# 学習過程の可視化
fig, ax = plt.subplots()


def animate(i):
    ax.clear()
    ax.set_ylim(min(y.data) - 0.1, max(y.data) + 0.1)
    ax.set_xlim(min(x.data) - 0.1, max(x.data) + 0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x.data, y.data, c='blue')
    ax.plot(x.data, y_pred_history[i], c='red')
    ax.set_title(f"linear regression\niter:{i:03}  y = {W_history[i]:.3f} x + {b_history[i]:.3f} loss:{loss_history[i]:.3f}")
    return

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=iters,
    interval=100,
)

ani.save('step42_linear_regression.gif')