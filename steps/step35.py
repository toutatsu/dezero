if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name='x'
y.name='y'

for i in range(8):
    gx = y if i==0 else x.grad # xで微分する関数
    x.cleargrad()
    gx.backward(create_graph=True)

    gx = x.grad
    gx.name = 'gx' + str(i+1)

    plot_dot_graph(gx, verbose=False, to_file=f"step35_tanh_{i+1}.png")