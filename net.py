import numpy as np

class DimensionException(Exception):
    def __init(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __str__(self):
        return repr([self.dim1, self.dim2])
        
def convolve(patch_in, kernel):
    # TODO:
    # assert that output is dimension inW-(kernelW-1) X inH-(kernelH-1)

    dims_out = [patch_in.shape[dim] - kernel.shape[dim]-1 for dim in [0,1]]
    if any([d<0 for d in dims_out]):
        raise DimensionException(patch_in.shape, kernel.shape)

    out = np.empty(dims_out)

    return out
