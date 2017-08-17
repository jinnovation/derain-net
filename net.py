import numpy as np

class DimensionException(Exception):
    def __init(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __str__(self):
        return repr([self.dim1, self.dim2])
        
def convolve(patch_in, kernel):
    dims_out = [patch_in.shape[dim] - (kernel.shape[dim]-1) for dim in [0,1]]
    if any([d<=0 for d in dims_out]):
        raise DimensionException(patch_in.shape, kernel.shape)

    return np.array([[
        np.sum(
            patch_in[i:i+kernel.shape[0],j:j+kernel.shape[1]] * kernel
        ) for j in range(dims_out[1])
    ] for i in range(dims_out[0])])
