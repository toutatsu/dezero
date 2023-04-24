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

# モデル
model = MLP((hidden_size, 3))

# オプティマイザー
optimizer = optimizers.SGD(lr).setup(model)

# 学習
data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)
loss_history = []

for epoch in range(max_epoch):

    # データセットのインデックスのシャッフル
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ミニバッチの生成
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        # batch = [train_set[i] for i in batch_index]
        batch_x = np.array([train_set[i][0] for i in batch_index])
        batch_t = np.array([train_set[i][1] for i in batch_index])

        # 勾配の算出
        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        # パラメータの更新
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)
    
    # 各エポックで学習結果を出力
    avg_loss = sum_loss / data_size
    print(f"epoch {epoch + 1}, loss {avg_loss:.2f}")
    loss_history.append(avg_loss)