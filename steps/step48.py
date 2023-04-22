if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ハイパーパラメータ
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# データ読み込み
x, t, data_indices = dezero.datasets.get_spiral(train=True)

# モデル
model = MLP((hidden_size, 3))

# オプティマイザー
optimizer = optimizers.SGD(lr).setup(model)

# 決定境界可視化用
h = 0.01
x_min, x_max = x[:,0].min() - .1, x[:,0].max() + .1
y_min, y_max = x[:,1].min() - .1, x[:,1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
Zs = []

# 学習
data_size = len(x)
max_iter = math.ceil(data_size / batch_size)
loss_history = []

for epoch in range(max_epoch):

    # データセットのインデックスのシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ミニバッチの生成
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 勾配の算出
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        # パラメータの更新
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
    
    # 決定境界可視化用の推論
    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    Zs.append(Z)
    
    # 各エポックで学習結果を出力
    avg_loss = sum_loss / data_size
    print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
    loss_history.append(avg_loss)


# 学習過程の可視化
fig, axes = plt.subplots(ncols=2, figsize=(10,4))

# データセットの順番を戻す
x_ = x.copy()
t_ = t.copy()
x_[data_indices] = x
t_[data_indices] = t

def animate(i):
    # 決定境界の可視化
    axes[0].clear()
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlim(-1, 1)

    axes[0].contourf(xx, yy, Zs[i])

    for t in range(3):
        axes[0].scatter(
            x_[100*t:100*(t+1), 0],
            x_[100*t:100*(t+1), 1],
            color=['orange','blue','green'][t],
            marker=['o', 'x', '^'][t]
        )

    axes[0].set_title(f"dicision boundary")

    # 損失の可視化
    axes[1].clear()
    axes[1].set_ylim(0, 1.3)
    axes[1].set_xlim(0, max_epoch)
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].plot(range(i), loss_history[:i], color='blue')
    axes[1].set_title(f"epoch:{i:03} loss = {loss_history[i]:.3f}")
    return

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=max_epoch,
    interval=100,
)
ani.save('step48_spiral.gif')