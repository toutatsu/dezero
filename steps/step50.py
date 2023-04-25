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
train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_dataloader = dezero.Dataloader(train_set, batch_size, shuffle=True)
test_dataloader = dezero.Dataloader(test_set, batch_size, shuffle=False)

# モデル
model = MLP((hidden_size, 3))

# オプティマイザー
optimizer = optimizers.SGD(lr).setup(model)

# 決定境界可視化用
h = 0.01
x_min, x_max = test_set.data[:,0].min() - .1, test_set.data[:,0].max() + .1
y_min, y_max = test_set.data[:,1].min() - .1, test_set.data[:,1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
Zs = []
# 学習
loss_history = {'train':[], 'test':[]}
acc_history = {'train':[], 'test':[]}

for epoch in range(max_epoch):

    sum_loss, sum_acc = 0, 0

    for x, t in train_dataloader:

        # 勾配の算出
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        # パラメータの更新
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    # 決定境界可視化用の推論
    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    Zs.append(Z)
    
    # 各エポックで学習結果を出力
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    print(f"epoch {epoch + 1}, loss {avg_loss:.4f} accuracy {avg_acc:.4f}")
    loss_history['train'].append(avg_loss)
    acc_history['train'].append(avg_acc)


    # テストデータで評価
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_dataloader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    
    avg_loss = sum_loss / len(test_set)
    avg_acc = sum_acc / len(test_set)
    print(f"epoch {epoch + 1}, loss {avg_loss:.4f} accuracy {avg_acc:.4f}")
    loss_history['test'].append(avg_loss)
    acc_history['test'].append(avg_acc)


# 学習過程の可視化
fig, axes = plt.subplots(ncols=3, figsize=(15,4))

# データセットの順番を戻す
x = test_set.data.copy()
t = test_set.label.copy()
x_ = test_set.data.copy()
t_ = test_set.label.copy()
x_[test_set.indices] = x
t_[test_set.indices] = t

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
    axes[1].plot(range(i), loss_history['train'][:i], color='blue', label='train')
    axes[1].plot(range(i), loss_history['test'][:i], color='red', label='test')
    axes[1].set_title(f"test loss = {loss_history['test'][i]:.3f}")
    axes[1].legend()

    # 精度の可視化
    axes[2].clear()
    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(0, max_epoch)
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('accuracy')
    axes[2].plot(range(i), acc_history['train'][:i], color='blue', label='train')
    axes[2].plot(range(i), acc_history['test'][:i], color='red', label='test')
    axes[2].set_title(f"test accuracy = {acc_history['test'][i]:.3f}")
    axes[2].legend()
    return

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=max_epoch,
    interval=100,
)
ani.save('step50_spiral.gif')