"""
Animated Circles
=============

This is based on a toot by celestialmaze (https://mastodon.social/@cmzw):
https://mastodon.social/@cmzw/111170900844809390

.. tags:: animation
"""

import time
import numpy as np
import napari
from skimage.draw import disk
import random
from napari.qt import thread_worker

# Constants
WIDTH, HEIGHT = 512, 512
TARGET_COLOR = np.array([255, 255, 0])  # Yellow

class Circle:
    def __init__(self):
        self.x, self.y, self.radius, self.color = self.initialize_circle()
        self.has_reached_top = False
        self.lifetime = 0
    
    def initialize_circle(self):
        x_center = WIDTH // 2 + int(random.gauss(0, 30))
        y_center = HEIGHT
        radius = random.randint(5, 35)
        
        color_offset = np.array([random.randint(-50, 50) for _ in range(3)])
        color = np.clip(TARGET_COLOR + color_offset, 0, 255)
        return x_center, y_center, radius, color

    def update(self):
        self.y -= 2
        self.radius *= 0.99
        self.lifetime += 1

        if self.radius < 1:
            self.has_reached_top = True
        
        fade_factor = 1 - (self.lifetime / (5*HEIGHT))
        self.color = np.clip((self.color * fade_factor).astype(np.uint8), 0, 255)

@thread_worker
def animate_circles():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    circles = [Circle() for _ in range(10)]
    
    while True:
        new_image = np.copy(image)

        for circle in circles:
            circle.update()
            rr, cc = disk((circle.y, circle.x), circle.radius)
            valid_coords = (rr >= 0) & (rr < HEIGHT) & (cc >= 0) & (cc < WIDTH)
            rr, cc = rr[valid_coords], cc[valid_coords]
    
            new_image[rr, cc] = circle.color

        circles = [c for c in circles if not c.has_reached_top]
        if random.random() < 0.8:
            circles.append(Circle())
        yield new_image
        time.sleep(0.01)

viewer = napari.Viewer()

def update_image(image):
    if 'animated_circles' in viewer.layers:
        viewer.layers['animated_circles'].data = image
    else:
        viewer.add_image(image, name='animated_circles')

worker = animate_circles()
worker.yielded.connect(update_image)
worker.start()
