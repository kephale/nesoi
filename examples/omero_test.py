
import numpy as np
import idr
import napari

viewer = napari.Viewer()
# viewer.window.add_plugin_dock_widget('napari-omero')

# viewer.open("https://idr.openmicroscopy.org/webclient/?show=image-9822152")

conn = idr.connection(host='ws://idr.openmicroscopy.org/omero-ws', user='public', password='public')


image_id = 6001247

image = conn.getObject("Image", image_id)


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



data = load_numpy_array(image)

print('data loaded', data.shape)

viewer.add_image(data)

if __name__ == '__main__':
    napari.run()

