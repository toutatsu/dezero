# pythonコマンドで実行する場合，グローバル変数__file__を使ってファイルがある場所の親ディレクトリをパスに追加
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)