if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
# y = x.reshape(6)

y.backward(retain_grad=True)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
# y = x.T

y.backward()
print(x.grad)


x = Variable(np.random.rand(1, 2, 3, 4))
y = x.transpose(1, 0, 3, 2)
z = y * 1

z.backward()
print(x.grad)