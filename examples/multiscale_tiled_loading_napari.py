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

empty = np.zeros((11087, 2000))

layer = viewer.add_image(empty)

print(layer.contrast_limits)
# layer.contrast_limits_range = (0, 300_000)
# layer.contrast_limits = (0, 80_000)
# layer.contrast_limits = (40_000, 185_000)
# layer.con
layer.contrast_limits_range = (0, 1)
layer.contrast_limits = (0, 1)

# Yield after each chunk is fetched
@thread_worker
def animator():
    for scale in reversed(range(4)):
        array = arrays[scale]
        # This can be an oct-tree fetch
        for x in range(0, array.shape[0], chunk_size[0]):
            for y in range(0, array.shape[1], chunk_size[1]):
                # time.sleep(0.1)
                z = int(500 / (2 ** scale))
                # result.data[x:(x + chunk_size[0]), y:(y + chunk_size[1]), z].compute()
                real_array = np.asarray(array[x:(x + chunk_size[0]), y:(y + chunk_size[1]), z].compute())
                upscaled = resize(real_array, [el * 2 ** scale for el in real_array.shape])
                # Return upscaled coordinates, the scale, and chunk
                yield (x * 2 ** scale, y * 2 ** scale, z, scale, upscaled)

worker = animator()

# when screen is moved, start a new animator
# disconnect old consumor and stop previous animator


def on_yield(coord):
    x, y, z, scale, chunk = coord
    t = time.time()
    chunk_size = chunk.shape
    print(f"Yielding at {(x, y, z)} scale = {scale} , {chunk_size}")
    layer.data[x:(x + chunk_size[0]), y:(y + chunk_size[1])] = chunk
    t2 = time.time()
    layer.refresh()
    t3 = time.time()
    print(f"Frame update {t}, {t2}, {t3}")


# viewer.camera.zoom = 100
    
worker.yielded.connect(on_yield)

worker.start()
napari.run()

