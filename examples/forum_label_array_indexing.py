import napari
import numpy as np
import skimage
from skimage.filters import threshold_otsu

# The post https://forum.image.sc/t/labeled-data-is-scattered/69377
#   suggests a possible indexing bug in the visualization of labels.
#   It might be the case that we're overflowing the datatype of our
#   array iterator.

image_size = (1753, 584, 1780)

# image_size = (200, 200, 200)

print('generating image')
img = np.random.random(image_size)
print('img', img.shape)
# img[1:,:,:] = 0

mgrid_img = np.mgrid[:image_size[0], :image_size[1], :image_size[2]][1, :, :, :]
mgrid_img = mgrid_img / np.max(mgrid_img[:])

print('mgrid_img', mgrid_img.shape)

# threshold = 0.5
threshold = 0.65

img = img * mgrid_img
print('mult', img.shape)

print('thresholding image')
threshold = threshold_otsu(img)
print('labeling image')
labeled_img = skimage.measure.label(img > threshold, connectivity=1)
print('viewing labels')
viewer = napari.view_labels(labeled_img)


if __name__ == '__main__':
    napari.run()
