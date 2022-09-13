
from typing import Dict, List

import numpy as np
import dask.array as da
from dask import delayed

from napari.types import LayerData
from vispy.color import Colormap

from omero.cli import ProxyStringType
from omero.gateway import BlitzObjectWrapper, ImageWrapper
from omero.model import enums as omero_enums
from omero.model import IObject


def load_numpy_array(image):
    pixels = image.getPrimaryPixels()
    size_z = image.getSizeZ()
    size_c = image.getSizeC()
    size_t = image.getSizeT()
    size_y = image.getSizeY()
    size_x = image.getSizeX()
    z, t, c = 0, 0, 0  # first plane of the image

    zct_list = []
    for t in range(size_t):
        for c in range(size_c):  # all channels
            for z in range(size_z):  # get the Z-stack
                zct_list.append((z, c, t))

    # Load all the planes as YX numpy array
    planes = pixels.getPlanes(zct_list)
    s = "t:%s c:%s z:%s y:%s x:%s" % (size_t, size_c, size_z, size_y, size_x)
    print(s)
    print("Downloading image %s" % image.getName())
    all_planes = np.stack(list(planes))
    shape = (size_t, size_c, size_z, size_y, size_x)
    return np.reshape(all_planes, newshape=shape)

# ===== From napari-omero


PIXEL_TYPES = {
    omero_enums.PixelsTypeint8: np.int8,
    omero_enums.PixelsTypeuint8: np.uint8,
    omero_enums.PixelsTypeint16: np.int16,
    omero_enums.PixelsTypeuint16: np.uint16,
    omero_enums.PixelsTypeint32: np.int32,
    omero_enums.PixelsTypeuint32: np.uint32,
    omero_enums.PixelsTypefloat: np.float32,
    omero_enums.PixelsTypedouble: np.float64,
}


def lookup_obj(conn, iobj: IObject) -> BlitzObjectWrapper:
    """Find object of type by ID."""
    conn.SERVICE_OPTS.setOmeroGroup("-1")
    type_ = iobj.__class__.__name__.rstrip("I")
    obj = conn.getObject(type_, iobj.id)
    if not obj:
        raise NameError(f"No such {type_}: {iobj.id}")

    return obj


def load_image_wrapper(image: ImageWrapper) -> List[LayerData]:
    data = get_data_lazy(image)
    meta = get_omero_metadata(image)
    # contrast limits range ... not accessible from plugin interface
    # win_min = channel.getWindowMin()
    # win_max = channel.getWindowMax()
    return (data, meta)


def get_omero_metadata(image: ImageWrapper) -> Dict:
    """Get metadata from OMERO as a Dict to pass to napari."""
    channels = image.getChannels()

    colors = []
    for ch in channels:
        # use current rendering settings from OMERO
        color = ch.getColor().getRGB()
        color = [r / 256 for r in color]
        colors.append(Colormap([[0, 0, 0], color]))

    contrast_limits = [
        [ch.getWindowStart(), ch.getWindowEnd()] for ch in channels
    ]

    visibles = [ch.isActive() for ch in channels]
    names = [ch.getLabel() for ch in channels]

    scale = None
    # Setting z-scale causes issues with Z-slider.
    # See https://github.com/tlambert03/napari-omero/pull/15
    # if image.getSizeZ() > 1:
    #     size_x = image.getPixelSizeX()
    #     size_z = image.getPixelSizeZ()
    #     if size_x is not None and size_z is not None:
    #         scale = [1, size_z / size_x, 1, 1]

    return {
        'channel_axis': 1,
        'colormap': colors,
        'contrast_limits': contrast_limits,
        'name': names,
        'visible': visibles,
        'scale': scale,
    }


def get_data_lazy(image: ImageWrapper) -> da.Array:
    """Get 5D dask array, with delayed reading from OMERO image."""
    nt, nc, nz, ny, nx = [getattr(image, f'getSize{x}')() for x in 'TCZYX']

    # print('get_data_lazy', [nt, nc, nz, ny, nx])
    
    pixels = image.getPrimaryPixels()
    dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)
    get_plane = delayed(lambda idx: pixels.getPlane(*idx))

    def get_lazy_plane(zct):
        return da.from_delayed(get_plane(zct), shape=(ny, nx), dtype=dtype)

    # print('example plane ()', get_plane((0, 0, 0)).compute())
    
    # 5D stack: TCZXY
    t_stacks = []
    for t in range(nt):
        c_stacks = []
        for c in range(nc):
            z_stack = []
            for z in range(nz):
                z_stack.append(get_lazy_plane((z, c, t)))
            c_stacks.append(da.stack(z_stack))
        t_stacks.append(da.stack(c_stacks))
    return da.stack(t_stacks)

# =====


def load_numpy_array_lazy(id, conn=None):
    wrapper = lookup_obj(
        conn, ProxyStringType("Image")(f"Image:{id}")
        )
    return load_image_wrapper(wrapper)
