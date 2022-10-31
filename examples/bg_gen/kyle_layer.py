import napari
import numpy as np

img = np.zeros((2,2))

napari.view_image(img, name='Kyle')

napari.run()
