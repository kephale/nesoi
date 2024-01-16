import napari
import numpy as np
from scipy.ndimage import rotate
from qtpy.QtCore import QTimer
from skimage.draw import line_aa

# Constants
size = 800
t = 0

# Create a blank image
img = np.zeros((size, size), dtype=np.uint8)

# Recursive function to draw lines
def draw_a(x, y, d, img, angle=0):
    if d > 9:
        # Calculate the end point of the line
        end_x = x + np.cos(angle) * d
        end_y = y + np.sin(angle) * d
        
        # Ensure the end coordinates are within the image boundaries
        end_x = np.clip(end_x, 0, img.shape[1] - 1)
        end_y = np.clip(end_y, 0, img.shape[0] - 1)
        
        # Draw the line using anti-aliasing, ensuring all points are within bounds
        rr, cc, val = line_aa(int(y), int(x), int(end_y), int(end_x))
        valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
        img[rr[valid], cc[valid]] = (val[valid] * 255).astype(np.uint8)
        
        # Calculate new angle based on the current position and global t
        new_angle = np.arctan2(end_y - y, end_x - x)
        
        # Recursively draw the next lines
        draw_a(end_x, end_y, d * 3/4, img, new_angle + t)
        draw_a(end_x, end_y, d * 3/4, img, new_angle - t)


# Update function for the Napari viewer
def update_layer(layer, img):
    global t
    img.fill(0)  # Clear the image
    draw_a(size // 2, size // 2, 200, img)  # Start the recursion
    layer.data = img
    layer.refresh()
    t += 0.005

# Initialize Napari viewer
viewer = napari.Viewer()
layer = viewer.add_image(img.copy(), name='Recursive Pattern')

# Timer for updating the image layer
timer = QTimer()
timer.timeout.connect(lambda: update_layer(layer, img))
timer.start(20) # update every 50 ms

napari.run()
