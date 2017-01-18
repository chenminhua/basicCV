import homography
from scipy import ndimage
import numpy as np
from PIL import Image
from pylab import *

def image_in_image(im1,im2,tp):
    m, n = im1.shape[:2]
    fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    # compute affine transform and apply
    H = homography.Haffine_from_points(tp,fp)
    im1_t = ndimage.affine_transform(im1,H[:2,:2],
                    (H[0,2],H[1,2]),im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t

im1 = np.array(Image.open('rollingStones.png').convert('L'))
im2 = np.array(Image.open('tucker2.png').convert('L'))

tp = np.array([[0,140,140,0], [170,170,300,300], [1,1,1,1]])

im3 = image_in_image(im2, im1, tp)

figure()
gray()
imshow(im3)
show()
