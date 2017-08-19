from guided_filter.core.filters import FastGuidedFilter

def split(img):
    gf = FastGuidedFilter(img, 16, 2, 4)
    base = gf.filter(img)
    detail = img / 255.0 - base
    return (base, detail)
