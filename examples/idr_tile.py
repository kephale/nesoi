import idr
import napari

import numpy as np
import dask.array as da
from dask import delayed

# from .omero_utils import load_numpy_array_lazy, PIXEL_TYPES
from omero_utils import load_numpy_array_lazy, PIXEL_TYPES

viewer = napari.Viewer()

conn = idr.connection(host='ws://idr.openmicroscopy.org/omero-ws', user='public', password='public')

# image_id = 9846151
image_id = 9822151

# Tile tests

image = conn.getObject("Image", image_id)
nt, nc, nz, ny, nx = [getattr(image, f'getSize{x}')() for x in 'TCZYX']

print('get_data_lazy', [nt, nc, nz, ny, nx])

x, y = 0, 0
width, height = 10000, 10000
    
tile = (x, y, width, height)
    
pixels = image.getPrimaryPixels()
dtype = PIXEL_TYPES.get(pixels.getPixelsType().value, None)

@delayed
def get_tile(x,y):
    print(f"fetching tile {x}, {y}")
    conn = idr.connection(host='ws://idr.openmicroscopy.org/omero-ws', user='public', password='public')
    image = conn.getObject("Image", image_id)
    pixels = image.getPrimaryPixels()
    return pixels.getTile(0, 0, 0, (x, y, width, height))

    
def get_lazy_tile(x, y):
    return da.from_delayed(get_tile(x, y), shape=(width, height), dtype=dtype)

print('starting to fetch')

# 5D stack: TCZXY
t_stacks = []
for t in range(nt):
    c_stacks = []
    for c in range(nc):
        z_stack = []
        for z in range(nz):
            y_stack = []
            for y in range(0, ny, height):
                x_stack = []
                # ragged border
                if y + height > ny:
                    pass
                else:
                    for x in range(0, nx, width):
                        print((x, y, z, c, t))
                        if x + width > nx:
                            pass
                        else:
                            x_stack.append(get_lazy_tile(x, y))
                    y_stack.append(da.concatenate(x_stack, axis=0))
            z_stack.append(da.stack(y_stack))
        c_stacks.append(da.stack(z_stack))
    t_stacks.append(da.stack(c_stacks))

result = da.stack(t_stacks)

value = result.sum()
print('starting to take sum')
value.compute()

print(f"sum: {value}")

# tile_a = get_lazy_tile(0, 0)
# tile_b = get_lazy_tile(0, 1000)

# only works with a 2D tile
# tiles = da.from_delayed(get_tile, dtype=dtype, shape=(width, height))

# stack = da.map_blocks(get_tile, dtype=dtype, meta=np.array((), dtype=dtype))
# stack = da.map_blocks(get_tile, dtype=dtype, meta={'dtype':dtype})

if __name__ == '__main__':
    napari.run()
