from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray, rgb2grey, rgb2hsv
from imageio import imread
from skimage import io

import matplotlib.pyplot as plt


img = io.imread('F:\pycharm_python\Paper-PyTorch\data\ParisStreetView\\test.jpg')
img_gray = rgb2gray(img)
io.imshow(img)
io.show()
io.imshow(img_gray)
print(img_gray.shape)
print(img.shape)
io.show()


edge = canny(img_gray, sigma=5)
io.imshow(edge)
io.show()
print(edge.shape)