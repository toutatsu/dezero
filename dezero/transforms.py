import numpy as np


class Compose():
    def __init__(self, transforms=[]):
        self.transforms = transforms
    
    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)

        return img


# =============================================================================
# Transforms for NumPy ndarray
# =============================================================================
class Normalize():

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean)==1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std)==1 else len(self.std)
            mean = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        
        return (array - mean) / std


class Flatten():

    def __call__(self, array):
        return array.flatten()


class AsType():

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType