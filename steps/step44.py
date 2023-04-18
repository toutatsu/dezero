if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero.utils import plot_dot_graph

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
x = np.sort(x, axis=0)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)


# 重みの初期化
l1 = L.Linear(10)
l2 = L.Linear(1)

# ニューラルネットワークの推論
def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


# 計算グラフ
# x.name = 'x'
# y.name = 'y'
# l1.W.name = 'W1'
# l1.b.name = 'b1'
# l2.W.name = 'W2'
# l2.b.name = 'b2'

# plot_dot_graph(
#     F.mean_squared_error(y, predict(x)),
#     verbose=False,
#     to_file=f"step44_neuralnetwork_sine.png"
# )


# 学習
lr = 0.2
iters = 10000
y_pred_history = []
loss_history = []


for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

    y_pred_history.append(y_pred.data.copy())
    loss_history.append(loss.data.copy())


# 学習過程の可視化
# fig, ax = plt.subplots()

# def animate(i):
#     ax.clear()
#     ax.set_ylim(-1, 2)
#     ax.set_xlim(min(x.data) - 0.1, max(x.data) + 0.1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.scatter(x.data, y.data, c='blue')
#     ax.plot(x.data, y_pred_history[100 * i], c='red')
#     ax.scatter(x.data, y_pred_history[100 * i], c='red')
#     ax.set_title(f"neural network\niter:{100 * i:03} loss:{loss_history[100 * i]:.3f}")
#     return

# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     frames=100,
#     interval=100,
# )

# ani.save('step44_neuralnetwork_sine.gif')