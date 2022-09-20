import napari

viewer = napari.viewer.Viewer()

viewer.show()

from napari_turing import TuringViewer

my_widget = TuringViewer(viewer)

my_widget.controler.play_click()
my_widget.hide()

napari.run()

