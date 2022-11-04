import idr
import napari

import numpy as np
import dask.array as da
from dask import delayed

# from .omero_utils import load_numpy_array_lazy, PIXEL_TYPES
from omero_utils import load_numpy_array_lazy, PIXEL_TYPES

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

ny = int(ny / height) * height
nx = int(nx / width) * width

# ny = height * 2
# nx = width * 1

# 5D stack: TCZXY
t_stacks = []
for t in range(nt):
    c_stacks = []
    for c in range(nc):
        z_stack = []
        for z in range(nz):
            x_stack = []
            for x in range(0, nx, height):
                y_stack = []
                # ragged border
                if x + width > nx:
                    pass
                else:
                    for y in range(0, ny, width):
                        print((x, y, z, c, t))
                        if y + height > ny:
                            pass
                        else:
                            y_stack.append(get_lazy_tile(x, y))
                    x_stack.append(da.concatenate(y_stack, axis=0))
            z_stack.append(da.stack(x_stack))
        c_stacks.append(da.stack(z_stack))
    t_stacks.append(da.stack(c_stacks))

result = da.stack(t_stacks)
# result.compute()

# Doing this will trigger fetching all tiles
# value = result.sum()
# print('starting to take sum')
# value.compute()
# print(f"sum: {value}")

viewer = napari.Viewer()
viewer.show()

viewer.camera.center = (0.0, 51626.14114205004, 5029.208202210406)
viewer.camera.zoom = 0.14077092440711514
# viewer.camera.angles = (0.0, 0.0, 90.0), perspective=0.0, interactive=True)

# viewer.add_image(result)

layer = napari.layers.Image(result)
viewer.layers.append(layer)

if __name__ == '__main__':
    napari.run()
