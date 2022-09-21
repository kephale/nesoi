import napari
from napari_animation import Animation
from napari_stable_diffusion import StableDiffusionWidget

viewer = napari.viewer.Viewer()
# viewer.show()
animation = Animation(viewer)

print('Animation started, starting StableDiffusion')
my_widget = StableDiffusionWidget(viewer)

my_widget.gallery_size.setValue(9)
my_widget.prompt_textbox.setText('unicorn pirate')
# The last index should be a cuda GPU if one is available
my_widget.device_list.setCurrentIndex(my_widget.device_list.count() - 1)
print('Running StableDiffusion')
my_widget._on_click()
print('StableDiffusion done')

animation.capture_keyframe()
animation.animate('demo.mov', canvas_only=False)
