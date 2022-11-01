import napari

from napari_turing import TuringViewer

from napari_animation import Animation

viewer = napari.Viewer()
viewer.show()

animation = Animation(viewer)

tv = TuringViewer(viewer)
tv.controler.play_click()

viewer.dims.ndisplay = 3
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe()
viewer.camera.zoom = 2.4
animation.capture_keyframe()
viewer.camera.angles = (-7.0, 15.7, 62.4)
animation.capture_keyframe(steps=60)
viewer.camera.angles = (2.0, -24.4, -36.7)
animation.capture_keyframe(steps=60)
viewer.reset_view()
viewer.camera.angles = (0.0, 0.0, 90.0)
animation.capture_keyframe()
animation.animate('demo.mov', canvas_only=False)

# if __name__ == '__main__':
#     napari.run()
