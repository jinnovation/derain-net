import net

class Noop:
    def process(self, img):
        return img

class Reverse:
    def process(self, img):
        return img[::-1]

class Convolver:
    def __init__(self, *kernel_sets):
        self.kernel_sets = kernel_sets

    def process(self, img):
        for ks in self.kernel_sets:
            img = net.convolve(img, *ks)
        return img

# guidedFilter(img,img,20,100000)
