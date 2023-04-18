if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
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
I, H, O = 1, 10, 1
W1, b1 = Variable(0.01 * np.random.randn(I, H)), Variable(np.zeros(H))
W2, b2 = Variable(0.01 * np.random.randn(H, O)), Variable(np.zeros(O))


# ニューラルネットワークの推論
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


# 計算グラフ
x.name = 'x'
y.name = 'y'
W1.name = 'W1'
b1.name = 'b1'
W2.name = 'W2'
b2.name = 'b2'

plot_dot_graph(
    F.mean_squared_error(y, predict(x)),
    verbose=False,
    to_file=f"step43_neuralnetwork_sine.png"
)


# 学習
lr = 0.2
iters = 10000
y_pred_history = []
W1_history = []
b1_history = []
W2_history = []
b2_history = []
loss_history = []


for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)

    y_pred_history.append(y_pred.data.copy())
    W1_history.append(W1.data.copy())
    b1_history.append(b1.data.copy())
    W2_history.append(W2.data.copy())
    b2_history.append(b2.data.copy())
    loss_history.append(loss.data.copy())


# 学習過程の可視化
fig, ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set_ylim(-1, 2)
    ax.set_xlim(min(x.data) - 0.1, max(x.data) + 0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x.data, y.data, c='blue')
    ax.plot(x.data, y_pred_history[100 * i], c='red')
    ax.scatter(x.data, y_pred_history[100 * i], c='red')
    ax.set_title(f"neural network\niter:{100 * i:03} loss:{loss_history[100 * i]:.3f}")
    return

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=100,
    interval=100,
)
print(W1_history[0].shape)

ani.save('step43_neuralnetwork_sine.gif')