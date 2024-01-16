import napari
import numpy as np
from qtpy.QtCore import QTimer

# Based on https://twitter.com/hisadan/status/1697878658437054785?s=20

# Constants
n = 999999
size = 800
primes = []

# Prime number calculation (done once)
p = np.zeros(n, dtype=int)
for i in range(2, n):
    if p[i] == 0:
        primes.append(i)
        for j in range(i*2, n, i):
            p[j] = i

# Image generation function
def generate_image(t):
    img = np.zeros((size, size), dtype=np.uint8)
    for prime in primes:
        x = int(prime * np.sin(prime * t) / 99 + size // 2)
        y = int(prime * np.cos(prime * t) / 99 + size // 2)
        if 0 <= x < size and 0 <= y < size:
            img[y:y+2, x:x+2] = 255  # make each point 2x2 pixels
    return img

# Update function for the Napari viewer
def update_layer(layer, t):
    img = generate_image(t)
    layer.data = img
    layer.refresh()

# Initialize Napari viewer
viewer = napari.Viewer()
layer = viewer.add_image(np.zeros((size, size)), name='Primes')

# Timer and time variable for updating the image layer
t = 1e-4  # starting t value, adjust as needed for proper visualization speed

def on_timer():
    global t
    update_layer(layer, t)
    t += 1e-4  # update t value for the spiral, adjust as needed

timer = QTimer()
timer.timeout.connect(on_timer)
timer.start(20)  # update every 100 ms

napari.run()
