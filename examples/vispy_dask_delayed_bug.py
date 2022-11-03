import napari

import numpy as np
import dask.array as da
from dask import delayed


nt = 1
nc = 1
nz = 1
ny = 167424
nx = 79360

# Tile tests

x, y = 0, 0
width, height = 10000, 10000
    
tile = (x, y, width, height)

dtype = np.uint8

@delayed
def get_tile(x,y):
    print(f"fetching tile {x}, {y}")
    return np.zeros((width, height))

    
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
