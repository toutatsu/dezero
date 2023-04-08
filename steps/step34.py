if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
plt.plot(x.data, y.data, label="y=sin(x)")

for i in range(3):
    gx = y if i==0 else x.grad # xで微分する関数
    x.cleargrad()
    gx.backward(create_graph=True)
    plt.plot(x.data, x.grad.data.copy(), label="y"+"'"*(i+1))

plt.legend(loc='lower right')
plt.title('derivatives of sin(x)')
plt.savefig("step34_derivatives_of_sin.png")