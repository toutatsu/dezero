if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, Model
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.utils import plot_dot_graph


# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
x = np.sort(x, axis=0)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
iters = 10000
hidden_size = 10
y_pred_history = []
loss_history = []


model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)


for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

    y_pred_history.append(y_pred.data.copy())
    loss_history.append(loss.data.copy())