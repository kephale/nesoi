import heapq
import logging
import sys

import napari
import numpy as np
import toolz as tz

from napari.experimental._progressive_loading import (
    MultiScaleVirtualData, VirtualData, chunk_priority_2D, chunk_slices,
    get_chunk)
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset)

from napari.qt.threading import thread_worker
from napari.utils.events import Event


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


def should_render_scale(scale, viewer):
    layer_name = get_layer_name_for_scale(scale)
    layer = viewer.layers[layer_name]
    layer_scale = layer.scale

    # pixel_size = 2 * np.tan(viewer.camera.angles[-1] / 2) * dist / max(layer_scale)
    pixel_size = viewer.camera.zoom * max(layer_scale)

    # TODO max pixel_size chosen by eyeballing
    return (pixel_size >= 0.5) and (pixel_size <= 4)


@thread_worker
def render_sequence(
    interval, num_threads=1, visible_scales=[], data=None
):
    """A generator that yields multiscale chunk tuples from low to high resolution.

    Parameters
    ----------
    interval : tuple
        ND coordinates of the topleft bottomright coordinates of the
        current view
    full_shape : tuple
        shape of highest resolution array
    num_threads : int
        number of threads for multithreaded fetching
    max_scale : int
        this is used to constrain the number of scales that are rendered
    """
    LOGGER.info(
        f"render_sequence: inside with corner pixels {interval} with max_scale {visible_scales}"
    )

    for scale in reversed(range(len(data.arrays))):
        if visible_scales[scale]:
            vdata = data._data[scale]
            chunk_keys = chunk_slices(vdata, ndim=2)

            LOGGER.info("render_sequence: computing priority")
            chunk_queue = chunk_priority_2D(chunk_keys, interval, scale)

            LOGGER.info(
                f"render_sequence: {scale}, {vdata.shape} fetching {len(chunk_queue)} chunks"
            )

            # Loop over all chunks in the queue and fetch the data
            while chunk_queue:
                priority, chunk_slice = heapq.heappop(chunk_queue)
                real_array = get_chunk(chunk_slice, array=vdata.array)
                yield (
                    tuple(chunk_slice),
                    scale,
                    data._scale_factors[scale],
                    real_array,
                )


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
        # worker.await_workers()
        worker.await_workers(msecs=5000)

    # Find the corners of visible data in the highest resolution
    corner_pixels = viewer.layers[get_layer_name_for_scale(0)].corner_pixels

    top_left = np.max((corner_pixels,), axis=0)[0, :]
    bottom_right = np.min((corner_pixels,), axis=0)[1, :]

    # TODO we could add padding around top_left and bottom_right to account for future camera movement

    # Interval is not valid, we cannot render
    if not np.all(top_left < bottom_right):
        LOGGER.info(
            f"dims_update_handler: interval is not valid {corner_pixels}"
        )
        return

    LOGGER.info(
        f"dims_update_handler: start render_sequence {corner_pixels} on layer {get_layer_name_for_scale(0)}"
    )

    # Find the visible scales
    visible_scales = [False] * len(data.arrays)

    for scale in range(len(data.arrays)):
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]
        layer_shape = layer.data.shape
        layer_scale = layer.scale

        layer.metadata["translated"] = False

        # Reenable visibility of layer
        visible_scales[scale] = should_render_scale(scale, viewer)

        layer.visible = visible_scales[scale]
        layer.opacity = 0.9

        pixel_size = viewer.camera.zoom * max(layer_scale)

        LOGGER.info(
            f"scale {scale} name {layer_name}\twith pixel_size {pixel_size}\ttranslate {layer.data.translate}"
        )

    # Update the MultiScaleVirtualData memory backing
    data.set_interval(top_left, bottom_right, visible_scales=visible_scales)

    # Start a new multiscale render
    worker = render_sequence(
        corner_pixels,
        data=data,
        visible_scales=visible_scales,
    )

    LOGGER.info("dims_update_handler: started render_sequence")

    # This will consume our chunks and update the numpy "canvas" and refresh
    def on_yield(coord):
        # TODO bad layer access
        chunk_slice, scale, scale_factors, chunk = coord
        layer_name = get_layer_name_for_scale(scale)
        layer = viewer.layers[layer_name]

        LOGGER.info(
            f"Writing chunk with size {chunk.shape} to: {(scale, (chunk_slice[0].start, chunk_slice[0].stop), (chunk_slice[1].start, chunk_slice[1].stop))} in layer {scale} with shape {layer.data.shape} and dataplane shape {layer.data.data_plane.shape} sum {chunk.sum()}"
        )
        # TODO hard coded scale factor
        if not layer.metadata["translated"]:
            layer.translate = np.array(layer.data.translate) * scale_factors

            # Toggle visibility of lower res layer
            if layer.metadata["prev_layer"]:
                layer.metadata["prev_layer"].opacity = 0.5
                # layer.metadata["prev_layer"].visible = False

        LOGGER.info("starting set_offset")
        # TODO check out set_offset and refresh are blocking
        layer.data.set_offset(chunk_slice, chunk)

        LOGGER.info("done with set_offset of chunk")

    worker.yielded.connect(on_yield)

    worker.start()


def add_progressive_loading_image(img, viewer=None):
    multiscale_data = MultiScaleVirtualData(img)

    LOGGER.info(f"MultiscaleData {multiscale_data.shape}")

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

    for scale, vdata in list(enumerate(multiscale_data._data)):
        # TODO scale is assumed to be powers of 2
        layer = viewer.add_image(
            vdata,
            contrast_limits=[0, 255],
            name=get_layer_name_for_scale(scale),
            scale=multiscale_data._scale_factors[scale],
            colormap='PiYG',
        )
        layers[scale] = layer
        layer.metadata["translated"] = False

    # Linked list of layers for visibility control
    for scale in reversed(range(len(layers))):
        layers[scale].metadata["prev_layer"] = (
            layers[scale + 1]
            if scale < len(multiscale_data._data) - 1
            else None
        )

    # TODO initial zoom should not be hardcoded
    # for mandelbrot scales=8
    # viewer.camera.zoom = 0.001
    viewer.camera.zoom = 0.00001
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

    import yappi
    
    def start_yappi():
        

        yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        yappi.start()

    # large_image = openorganelle_mouse_kidney_em()
    large_image = mandelbrot_dataset()

    multiscale_img = large_image["arrays"]
    viewer._layer_slicer._force_sync = False

    rendering_mode = "progressive_loading"

    if rendering_mode == "progressive_loading":
        # Make an object that creates/manages all scale nodes
        add_progressive_loading_image(multiscale_img, viewer=viewer)
    else:
        layer = viewer.add_image(multiscale_img)

    def stop_yappi():
        yappi.stop()

        yappi.get_func_stats().print_all()
        yappi.get_thread_stats().print_all()


    def yappi_stats():
        thread_stats = yappi.get_thread_stats()
