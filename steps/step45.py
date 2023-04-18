if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP
from dezero.utils import plot_dot_graph

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
x = np.sort(x, axis=0)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)


# モデルの定義
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = self.l2(F.sigmoid(self.l1(x)))
        return y


# 学習
lr = 0.2
iters = 10000
hidden_size = 10
y_pred_history = []
loss_history = []

# model = TwoLayerNet(hidden_size, out_size=1)
model = MLP((10, 1))
model.plot(x, to_file='step45_twolayernet.png')

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
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

# ani.save('step45_twolayernet_sine.gif')