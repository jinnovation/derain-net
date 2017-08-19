import numpy as np

class DimensionException(Exception):
    def __init(self, shape_patch=None, shapes_kernel=None):
        self.shape_patch = shape_patch
        self.shapes_kernel= shapes_kernel

    def __str__(self):
        return repr([self.shape_patch, self.shapes_kernel])
    
def convolve(patch_in, *kernels):
    shapes_kernel = [k.shape for k in kernels]
    if len(set(shapes_kernel)) != 1:
        raise DimensionException(None, shapes_kernel)

    k_shape = kernels[0].shape

    shape_out = [patch_in.shape[dim] - (k_shape[dim]-1) for dim in [0,1]]
    if (any([d<=0 for d in shape_out])
        or any(shape_out[d] > patch_in.shape[d] for d in [0,1])
        or patch_in.shape[2] != k_shape[2]
    ):
        raise DimensionException(patch_in.shape, shapes_kernel)

    return np.concatenate([
        [[[np.sum(patch_in[i:i+kernel.shape[0],j:j+kernel.shape[1]] * kernel)
        ] for j in range(shape_out[1])
        ] for i in range(shape_out[0])
        ] for kernel in kernels], axis=2)
