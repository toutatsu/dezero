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
max_epoch = 5
batch_size = 100
hidden_size = 1000

# データ読み込み
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_dataloader = dezero.Dataloader(train_set, batch_size, shuffle=True)
test_dataloader = dezero.Dataloader(test_set, batch_size, shuffle=False)

# モデル
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)

# オプティマイザー
optimizer = optimizers.Adam().setup(model)

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

    print(f"epoch : {epoch + 1}")
    # 各エポックで学習結果を出力
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    print(f"loss {avg_loss:.4f} accuracy {avg_acc:.4f}")
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
    print(f"loss {avg_loss:.4f} accuracy {avg_acc:.4f}")
    loss_history['test'].append(avg_loss)
    acc_history['test'].append(avg_acc)