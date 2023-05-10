import logging
import sys
from typing import Tuple, Union

import dask.array as da
import napari
import numpy as np
import toolz as tz
import zarr
from napari.experimental._progressive_loading import (
    MultiScaleVirtualData, VirtualData, get_chunk, interpolated_get_chunk_2D)
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset, openorganelle_mouse_kidney_em)
from napari.layers._data_protocols import Index, LayerDataProtocol
from napari.qt.threading import thread_worker
from napari.utils.events import Event
from numba import njit
from numcodecs import Blosc
from psygnal import debounced
from skimage.transform import resize
from superqt import ensure_main_thread
from zarr.storage import init_array, init_group
from zarr.util import json_dumps

# config.async_loading = True

LOGGER = logging.getLogger("mandelbrot_vizarr")
LOGGER.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(streamHandler)

# This global worker is used for fetching data
global worker
worker = None

"""
Current differences between this (2D) and are_the_chunks_in_view (3D):
- 2D v 3D
- 2D does not use chunk prioritization
- 2D uses linear interpolation
"""


def chunks_for_scale(corner_pixels, array, scale):
    """Return the keys for all chunks at this scale within the corner_pixels

    Parameters
    ----------
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    array : arraylike
        a ND numpy array with the data for this scale
    scale : int
        the scale level, assuming powers of 2

    """

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code

    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    chunk_size = array.chunksize

    # TODO kludge for 3D z-only interpolation
    # zval = mins[-3]

    # Find the extent from the current corner pixels, limit by data shape
    # TODO risky int cast
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(np.uint64)
    maxs = np.min(
        (np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)),
        axis=0,
    ).astype(np.uint64)

    # mins[-3] = maxs[-3] = zval

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])
    # zs = [zval]
    # TODO kludge

    for x in xs:
        for y in ys:
            yield (
                slice(y, (y + chunk_size[-2])),
                slice(x, (x + chunk_size[-1])),
            )


def get_and_process_chunk_2D(
    chunk_slice,
    scale,
    array,
    full_shape,
):
    """Fetch and upscale a chunk

    Parameters
    ----------
    chunk_slice : tuple of slices
        a key corresponding to the chunk to fetch
    scale : int
        scale level, assumes power of 2
    array : arraylike
        the ND array to fetch a chunk from
    full_shape : tuple
        a tuple storing the shape of the highest resolution level

    """
    # Trigger a fetch of the data
    real_array = interpolated_get_chunk_2D(
        chunk_slice,
        array=array,
    )

    # TODO imposes 2D
    y, x = [sl.start for sl in chunk_slice]

    LOGGER.info(
        f"\tyield will be placed at: {(y * 2**scale, x * 2**scale, scale, real_array.shape)} slice: {(chunk_slice[0].start, chunk_slice[0].stop, chunk_slice[0].step)} {(chunk_slice[1].start, chunk_slice[1].stop, chunk_slice[1].step)}"
    )

    return (
        tuple(chunk_slice),
        scale,
        real_array,
    )


@thread_worker
def render_sequence(corner_pixels, num_threads=1, max_scale=1, data=None):
    """A generator that yields multiscale chunk tuples from low to high resolution.

    Parameters
    ----------
    corner_pixels : tuple
        ND coordinates of the topleft bottomright coordinates of the
        current view
    full_shape : tuple
        shape of highest resolution array
    num_threads : int
        number of threads for multithreaded fetching
    max_scale : int
        this is used to constrain the number of scales that are rendered
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    LOGGER.info(
        f"render_sequence: inside with corner pixels {corner_pixels} with max_scale {max_scale}"
    )

    for scale in reversed(range(len(data.arrays))):
        # TODO hard coded usage of large_image
        if scale >= max_scale:
            array = data.arrays[scale]

            chunks_to_fetch = list(
                chunks_for_scale(corner_pixels, array, scale)
            )

            # if corner_pixels[0,0] > 0:
            #     import pdb; pdb.set_trace()

            # TODO pickup here, virtualdata.translate is weird, still getting (0,0) chunks

            LOGGER.info(
                f"render_sequence: {scale}, {array.shape} fetching {len(chunks_to_fetch)} chunks"
            )

            if num_threads == 1:
                # Single threaded:
                for chunk_slice in chunks_to_fetch:
                    yield get_and_process_chunk_2D(
                        chunk_slice,
                        scale,
                        array,
                        data.shape,
                    )
            else:
                raise Exception("Multithreading was removed")


def get_layer_name_for_scale(scale):
    return f"scale_{scale}"


@tz.curry
def dims_update_handler(invar, data=None):
    """Start a new render sequence with the current viewer state

    Parameters
    ----------
    invar : Event or viewer
        either an event or a viewer
    full_shape : tuple
        a tuple representing the shape of the highest resolution array

    """
    global worker, viewer

    LOGGER.info("dims_update_handler")

    # This function can be triggered 2 different ways, one way gives us an Event
    if type(invar) is not Event:
        viewer = invar

    # TODO global worker usage is not viable for real implementation
    # Terminate existing multiscale render pass
    if worker:
        # TODO this might not terminate threads properly
        worker.quit()

    # Find the corners of visible data in the canvas
    corner_pixels = viewer.layers[get_layer_name_for_scale(0)].corner_pixels

    # Shouldn't need to use canvas_corners at all
    # canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
    #     np.uint64
    # )

    # Offset corner_pixels by data offset
    # corner_pixels = np.array([corner + offset for corner, offset in zip(corner_pixels, viewer.layers[get_layer_name_for_scale(0)].data.translate)])
    
    top_left = np.max((corner_pixels, ), axis=0)[0, :]
    bottom_right = np.min((corner_pixels, ), axis=0)[1, :]

    # TODO we could add padding around top_left and bottom_right to account for future camera movement

    # Interval must be nonnegative
    if not np.all(top_left < bottom_right):
        import pdb; pdb.set_trace()
       

    # TODO Image.corner_pixels behaves oddly maybe b/c VirtualData
    # if bottom_right.shape[0] > 2:
    #     bottom_right[0] = canvas_corners[1, 0]

    corners = np.array([top_left, bottom_right], dtype=np.uint64)

    LOGGER.info("dims_update_handler: start render_sequence")

    # Find the maximum scale to render
    max_scale = len(data.arrays) - 1
    
    for scale, layer in enumerate(viewer.layers):
        layer_shape = layer.data.shape
        layer_scale = layer.scale

        layer.metadata["translated"] = False
        # Reenable visibility of layer
        # layer.visible = True
        layer.opacity = 1.0

        scaled_shape = [sh * sc for sh, sc in zip(layer_shape, layer_scale)]

        # TODO this dist calculation assumes orientation
        # dist = sum([(t - c)**2 for t, c in zip(layer.translate, viewer.camera.center[1:])]) ** 0.5

        # pixel_size = 2 * np.tan(viewer.camera.angles[-1] / 2) * dist / max(layer_scale)
        pixel_size = viewer.camera.zoom * max(layer_scale)

        print(f"scale {scale}\twith pixel_size {pixel_size}\ttranslate {layer.data.translate}")
        if pixel_size > 1:
            max_scale = min(max_scale, scale)

    # Update the MultiScaleVirtualData memory backing
    data.set_interval(top_left, bottom_right, max_scale=max_scale)
            
    # Start a new multiscale render
    worker = render_sequence(
        corners,
        data=data,
        max_scale=max_scale,
    )

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        # TODO bad layer access
        chunk_slice, scale, chunk = coord
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]
        # chunk_size = chunk.shape
        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: {(scale, (chunk_slice[0].start, chunk_slice[0].stop), (chunk_slice[1].start, chunk_slice[1].stop))}"
        )
        # TODO hard coded scale factor
        if not layer.metadata["translated"]:            
            layer.translate = np.array(layer.data.translate) * 2 ** scale

            # Toggle visibility of lower res layer
            if layer.metadata["prev_layer"]:
                layer.metadata["prev_layer"].opacity = 0.5
            #     layer.metadata["prev_layer"].visible = False
            

        layer.data.set_offset(chunk_slice, chunk)
        # layer.data[chunk_slice] = chunk    
        
        layer.refresh()

    worker.yielded.connect(on_yield)

    worker.start()


def add_progressive_loading_image(img, viewer=None):

    multiscale_data = MultiScaleVirtualData(img)
    
    LOGGER.info(
        f"MultiscaleData {multiscale_data.shape}"
    )

    # Get initial extent for rendering
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        np.uint64
    )
    top_left = canvas_corners[0, :]
    bottom_right = canvas_corners[1, :]

    multiscale_data.set_interval(top_left, bottom_right)

    # TODO sketchy Disable _update_thumbnail
    def temp_update_thumbnail(self):
        self.thumbnail = np.ones((32, 32, 4))
        
    napari.layers.image.Image._update_thumbnail = temp_update_thumbnail
    
    # We need to initialize the extent of each VirtualData
    layers = {}
    # Start from back to start because we build a linked list
    
    for scale, vdata in reversed(list(enumerate(multiscale_data._data))):

        # TODO scale is assumed to be powers of 2
        layer = viewer.add_image(
            vdata,
            contrast_limits=[0, 255],
            name=get_layer_name_for_scale(scale),
            scale=multiscale_data._scale_factors[scale],
        )
        layers[scale] = layer
        layer.metadata["translated"] = False
        # Linked list of layers
        layer.metadata["prev_layer"] = layers[scale + 1] if scale < len(multiscale_data._data) - 1 else None

    # TODO initial zoom should not be hardcoded
    viewer.camera.zoom = 0.001
    canvas_corners = viewer.window.qt_viewer._canvas_corners_in_world.astype(
        np.uint64
    )
    LOGGER.info(f"viewer canvas corners {canvas_corners}")

    # Connect to camera and dims
    for listener in [viewer.camera.events, viewer.dims.events]:
        listener.connect(
            debounced(
                ensure_main_thread(dims_update_handler(data=multiscale_data)),
                timeout=1000,
            )
        )

    # Trigger first render
    dims_update_handler(viewer, data=multiscale_data)


if __name__ == "__main__":
    global viewer
    viewer = napari.Viewer()

    # large_image = openorganelle_mouse_kidney_em()
    large_image = mandelbrot_dataset()

    multiscale_img = large_image["arrays"]
    viewer._layer_slicer._force_sync = True

    rendering_mode = "progressive_loading"

    if rendering_mode == "progressive_loading":
        # Make an object that creates/manages all scale nodes
        add_progressive_loading_image(multiscale_img, viewer=viewer)
    else:
        layer = viewer.add_image(multiscale_img)
