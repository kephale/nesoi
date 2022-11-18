import h5py
import numpy as np
import imglyb
import scyjava
import threading
import time
import jpype
from jpype import JImplements, JOverride, JArray, JString

# ----- Java dependencies

scyjava.config.add_repositories({'jitpack.io': 'https://jitpack.io'})

scyjava.config.endpoints.append('net.imglib2:imglib2:5.12.0')

scyjava.start_jvm()

# Block until JVM is started
while not scyjava.jvm_started():
    time.sleep(0.02)

# ----- Java imports

ArrayList = scyjava.jimport('java.util.ArrayList')

Views = scyjava.jimport('net.imglib2.view.Views')
RandomAccessibleInterval = scyjava.jimport('net.imglib2.RandomAccessibleInterval')
Intervals = scyjava.jimport('net.imglib2.util.Intervals')

# AffineTransform3D = scyjava.jimport('net.imglib2.realtransform.AffineTransform3D')

Callable = scyjava.jimport('java.util.concurrent.Callable')
ForkJoinPool = scyjava.jimport('java.util.concurrent.ForkJoinPool')
ForkJoinTask = scyjava.jimport('java.util.concurrent.ForkJoinTask')

File = scyjava.jimport('java.io.File')
ImageIO = scyjava.jimport('javax.imageio.ImageIO')

UnsignedByteType = scyjava.jimport('net.imglib2.type.numeric.integer.UnsignedByteType')

# ------ 

width = 800
height = 600

render_image_ndarray = np.zeros((width, height), dtype=np.dtype(int))

render_image_jarray = JArray.of(render_image_ndarray)

# This draws with Python
def draw(offset=0):
    for x in range(width):
        render_image_ndarray[x, :] = (x + offset) % width
    return render_image_ndarray

draw(offset=0)

from napari.qt.threading import thread_worker

import napari

viewer = napari.Viewer()

layer = viewer.add_image(render_image_ndarray)

@thread_worker
def animator():
    while True:
        time.sleep(0.1)
        yield time.time()
        
        print('asdfdsaf')
        
worker = animator()

def on_yield(t):
    render_image_ndarray = draw(offset=int(t))
    layer.data = render_image_ndarray
    layer.refresh()


worker.yielded.connect(on_yield)

worker.start()
napari.run()

