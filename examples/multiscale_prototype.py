import numpy as np
import threading
import time

import zarr

import dask
import dask.array as da

import logging

logging.basicConfig(encoding='utf-8', level=logging.WARNING)

import s3fs

# width = 10_000
# height = 10_000

# value_shape = (width, height)
# value = dask.delayed(np.ones)(value_shape)
# dask_array = da.from_delayed(value, value_shape, dtype=float)

from napari.qt.threading import thread_worker

import napari

viewer = napari.Viewer()

from fibsem_tools.io import read_xarray

from skimage.transform import resize

# TODO open the janelia image
container = "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5"
dataset = "em/fibsem-uint16"

scale_levels = range(4)

# uri = 's3://janelia-cosem-datasets/jrc_sum159-1/jrc_sum159-1.n5/labels/gt/0003/crop26/labels/all/s0/'
uri = f"{container}/{dataset}/s0/"

arrays = [read_xarray(f"{container}/{dataset}/s{scale}/", storage_options={'anon' : True}) for scale in scale_levels]

# Yield copies of the dask array with columns computed one by one

# dask_array = result.data[1000:1020, 1000:1020, 5000]

chunk_size = (384, 384, 384)

empty = np.zeros((11087, 10000))

layer = viewer.add_image(empty)

print(layer.contrast_limits)
# layer.contrast_limits_range = (0, 300_000)
# layer.contrast_limits = (0, 80_000)
# layer.contrast_limits = (40_000, 185_000)
# layer.con
layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)

from itertools import count

# Yield after each chunk is fetched
@thread_worker
def animator(corner_pixels):
    
    for scale in reversed(range(4)):
        array = arrays[scale]

        # there is a per-scale offset
        
        # Scale corner pixels
        y1, x1 = corner_pixels[0,:] / (2 ** scale)
        y2, x2 = corner_pixels[1,:] / (2 ** scale)

        y1 = int(np.floor(y1 / chunk_size[0]) * chunk_size[0])
        x1 = int(np.floor(x1 / chunk_size[1]) * chunk_size[1])
        y2 = min(int(np.ceil(y2 / chunk_size[0]) * chunk_size[0]), array.shape[0])
        x2 = min(int(np.ceil(x2 / chunk_size[1]) * chunk_size[1]), array.shape[1])

        xs = range(x1, x2, chunk_size[1])
        ys = range(y1, y2, chunk_size[0])

        print(f"scale coords: {xs}\n {ys}")
        
        print(f"animator {array.shape} {scale} {(x1, y1, x2, y2)}")
        
        for x in xs:
            for y in ys:
                # time.sleep(0.1)
                z = int(150 / (2 ** scale))
                print(f"animator {x} {y} {z}, {chunk_size}, {array.shape}")
                # result.data[x:(x + chunk_size[0]), y:(y + chunk_size[1]), z].compute()
                real_array = np.asarray(array[y:(y + chunk_size[0]), x:(x + chunk_size[1]), z].compute())
                upscaled = resize(real_array, [el * 2 ** scale for el in real_array.shape])
                # Return upscaled coordinates, the scale, and chunk
                yield (x * 2 ** scale, y * 2 ** scale, z, scale, upscaled)


global worker
worker = None
                
@viewer.bind_key('k')
def dims_update_handler(viewer):
    global worker

    if worker:
        print('stop previous worker')
        worker.quit()

    print('starting new worker')

    for array in arrays:
        print(f"array {array.shape}")
    
    # corner_pixels = layer.corner_pixels
    # data_corners = layer._transforms[1:].simplified.set_slice(sorted(layer._slice_input.displayed)).inverse(corner_pixels)

    # print(f"corners {corner_pixels}, {data_corners}")

    corner_pixels = layer.corner_pixels
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(int)

    print(corner_pixels, '=====', canvas_corners)
    top_left = (int(np.max((corner_pixels[0, 0], canvas_corners[0, 0]))),
                int(np.max((corner_pixels[0, 1], canvas_corners[0, 1]))))
    bottom_right = (int(np.min((corner_pixels[1, 0], canvas_corners[1, 0]))),
                    int(np.min((corner_pixels[1, 1], canvas_corners[1, 1]))))
    
    corners = np.array([top_left, bottom_right], dtype=int)    

    print(f"corners {corners}")
    
    worker = animator(corners)

    # when screen is moved, start a new animator
    # disconnect old consumor and stop previous animator


    def on_yield(coord):
        x, y, z, scale, chunk = coord
        t = time.time()
        chunk_size = chunk.shape
        print(f"Yielding at {(x, y, z)} scale = {scale} , {chunk_size}")
        layer.data[y:(y + chunk_size[0]), x:(x + chunk_size[1])] = chunk
        t2 = time.time()
        layer.refresh()
        t3 = time.time()
        print(f"Frame update {t}, {t2}, {t3}")

    def on_return(coord):
        print(f"Completed rendering of {coord}")
        
    # viewer.camera.zoom = 100

    worker.returned.connect(on_return)
    worker.yielded.connect(on_yield)

    worker.start()

# viewer.dims.events.current_step.connect(dims_update_handler)
    
napari.run()

