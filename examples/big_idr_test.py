import idr
import napari

from .omero_utils import load_numpy_array_lazy

viewer = napari.Viewer()


conn = idr.connection(host='ws://idr.openmicroscopy.org/omero-ws', user='public', password='public')


image_id = 9846151

# napari.layers.Image._is_async = lambda x: False

# data = load_numpy_array(image)
data, meta = load_numpy_array_lazy(image_id, conn=conn)

# image = conn.getObject("Image", image_id)

print('data loaded', data.shape)

print('async', napari.layers.Image._is_async(None))

viewer.add_image(data)

print('image added to viewer')

if __name__ == '__main__':
    napari.run()

