class Noop:
    def process(self, img):
        return img

class Reverse:
    def process(self, img):
        return img[::-1]

# guidedFilter(img,img,20,100000)
