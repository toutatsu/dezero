from dezero import Variable
import numpy as np


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(f, x):

    """
    勾配確認
    """

    num_grad = numerical_grad(f, x)
    y = f(x)
    y.backward()
    bp_grad = y.grad

    assert bp_grad.shape == num_grad.shape
    
    res = array_allclose(num_grad, bp_grad)

    if not res:
        print('')
        print('='*10 + ' FAILED (Gradient Check) '+ '='*10)
        print('Numerical Grad')
        print(f'shape: {num_grad.shape}')
        print(f'values: {str(num_grad.flatten()[:10])}')
        print('Backprop Grad')
        print(f'shape: {bp_grad.shape}')
        print(f'values: {str(bp_grad.flatten()[:10])}')
    
    return res


def numerical_grad(f, x):
    
    """
    数値微分
    """
    
    eps = 1e-4

    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (eps * 2)


def array_equal(a, b):

    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    return np.array_equal(a, b)


def array_allclose(a, b, rtol=1e-4, atol=1e-5):

    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b

    return np.allclose(a, b, rtol=rtol, atol=atol)