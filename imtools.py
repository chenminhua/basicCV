from PIL import Image
from pylab import *

def histeq(im, bins=256):
  imhist, bins = histogram(im.flatten(), bins, normed=True)
  cdf = imhist.cumsum()
  cdf = 255 * cdf / cdf[-1]
  im2 = interp(im.flatten(), bins[:-1], cdf)
  return im2.reshape(im.shape), cdf
