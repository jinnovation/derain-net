class DimensionException(Exception):
    def __init(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __str__(self):
        return repr([self.dim1, self.dim2])
        
def convolve(patch_in, kernel):
    # TODO:
    # assert that output is dimension inW-(kernelW-1) X inH-(kernelH-1)

    patch_out = patch_in
    return patch_out
