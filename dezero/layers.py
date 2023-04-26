import os
from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
from dezero import cuda

# =============================================================================
# Layer (base class)
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set() # パラメータの集合


    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)): # Parameterのインスタンス変数のみ_paramsに追加
            self._params.add(name)
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

    
    def forward(self, inputs):
        raise NotImplementedError()


    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer): # Layerインスタンスからパラメータを取り出す
                yield from obj.params()
            else:
                yield obj


    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    # LayerのもつParameterの入れ子でない辞書 param_dictを取得
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


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


    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
        

    def forward(self, x):
        # データを流すタイミングで重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y