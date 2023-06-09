import os
import subprocess
import urllib.request
import numpy as np
from dezero import Variable


# =============================================================================
# Visualize for computational graph
# =============================================================================


def _dot_var(v, verbose=False):
    """
    v : Variable インスタンス
    DOT言語のノード情報をもつ文字列を返す
    """

    name = '' if v.name is None else v.name

    if verbose and v.data is not None:

        if v.name is not None:
            name += ': '
        
        name += str(v.shape) + ' ' + str(v.dtype)

    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f):
    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'

    for x in f.inputs:
        txt += f"{id(x)} -> {id(f)}\n"
    for y in f.outputs:
        txt += f"{id(f)} -> {id(y())}\n"
    
    return txt


def get_dot_graph(output, verbose=False):

    txt = ''

    # 以下 Variable.backwardと同様
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
            # funcs.sort(key=lambda x: x.generation) #ノードの順番は関係なし

    add_func(output.creator)
    txt += _dot_var(output, verbose) # 出力変数の情報

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f) # 関数の情報

        for x in f.inputs:
            txt += _dot_var(x, verbose) # 関数へ入力される変数の情報

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file='graph.png'):

    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    # ~/.dezero ディレクトリを作成
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # dotコマンドを実行
    extension = os.path.splitext(to_file)[1][1:]
    subprocess.run(f'dot {graph_path} -T {extension} -o {to_file}', shell=True)


    # Jupyter Notebook用に画像を表示
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================


# TODO: 動作を理解する
def sum_to(x, shape):

    lead = x.ndim - len(shape)
    lead_axis = tuple(range(lead))

    y = x.sum(lead_axis + tuple([i + lead for i, sx in enumerate(shape) if sx == 1]), keepdims = True)

    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


# TODO: 動作を理解する
def reshape_sum_backward(gy, x_shape, axis, keepdims):

    ndim = len(x_shape)

    tupled_axis = axis

    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >=0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy


# TODO: 動作を理解する
def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m

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


# =============================================================================
# download function
# =============================================================================


def show_progress(block_num, block_size, total_size):

    downloaded = block_num * block_size
    p = downloaded / total_size * 100 # パーセンテージ
    i = int(downloaded / total_size * 30) # プログレスバーの進度
    if p >= 100: p = 100
    if i >= 30: i = 30
    print(f"\r[{'#' * i + '.' * (30 - i)}] {p:.2f}%", end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url, file_name=None):

    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path
    
    print('Downloading: ' + file_name)

    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except(Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(' Done')

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError