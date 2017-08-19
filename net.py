import numpy as np

class DimensionException(Exception):
    def __init(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __str__(self):
        return repr([self.dim1, self.dim2])
    
def convolve(patch_in, *kernels):
    if len(set([k.shape for k in kernels])) != 1:
        raise DimensionException(kernels[0].shape)

    k_shape = kernels[0].shape

    dims_out = [patch_in.shape[dim] - (k_shape[dim]-1) for dim in [0,1]]
    if (any([d<=0 for d in dims_out])
        or any(dims_out[d] > patch_in.shape[d] for d in [0,1])
        or patch_in.shape[2] != k_shape[2]
    ):
        raise DimensionException(patch_in.shape, k_shape)

    return np.concatenate([np.array(
        [[[np.sum(
            patch_in[i:i+kernel.shape[0],j:j+kernel.shape[1]] * kernel
        )] for j in range(dims_out[1])] for i in range(dims_out[0])]) for kernel in kernels],
                          axis=2)
