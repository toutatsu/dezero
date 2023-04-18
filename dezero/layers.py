from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F

# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self.__params = set() # パラメータの集合


    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)): # Parameterのインスタンス変数のみ__paramsに追加
            self.__params.add(name)
        super().__setattr__(name, value)


    def __call__(self, *inputs):

        outputs = self.forward(*inputs)

        # Function.__call__同様 タプルではない場合の追加対応
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
    
        # 入出力変数を弱参照で保持
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    
    def forward(self, *inputs):
        raise NotImplementedError


    def params(self):
        for name in self.__params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer): # Layerインスタンスからパラメータを取り出す
                yield from obj.params()
            else:
                yield obj


    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


# =============================================================================
# Linear
# =============================================================================
class Linear(Layer):

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_size is not None: # in_sizeが指定されていない場合はforward()内で用意
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')


    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        

    def forward(self, x):
        # データを流すタイミングで重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y