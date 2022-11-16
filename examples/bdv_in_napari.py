
import h5py
import numpy as np
import imglyb
import scyjava
import threading
import time
import jpype
from jpype import JImplements, JOverride, JArray, JString

# ----- Java dependencies

scyjava.config.add_repositories({'jitpack.io': 'https://jitpack.io'})

scyjava.config.endpoints.append('net.imglib2:imglib2:5.12.0')
scyjava.config.endpoints.append('org.scijava:ui-behaviour:2.0.7')
# scyjava.config.endpoints.append('sc.fiji:bigdataviewer-core:10.4.5')
scyjava.config.endpoints.append('com.github.kephale:bigdataviewer-core:4384d69')
# scyjava.config.endpoints.append('sc.fiji:bigdataviewer-vistools:1.0.0-beta-31')
scyjava.config.endpoints.append('com.github.kephale:bigdataviewer-vistools:5855f7d')

scyjava.start_jvm()

# Block until JVM is started
while not scyjava.jvm_started():
    time.sleep(0.02)

# ----- Java imports

ArrayList = scyjava.jimport('java.util.ArrayList')

Views = scyjava.jimport('net.imglib2.view.Views')
RandomAccessibleInterval = scyjava.jimport('net.imglib2.RandomAccessibleInterval')
Intervals = scyjava.jimport('net.imglib2.util.Intervals')

BigDataViewer = scyjava.jimport('bdv.BigDataViewer')
SourceAndConverter = scyjava.jimport('bdv.viewer.SourceAndConverter')

BdvFunctions        = scyjava.jimport('bdv.util.BdvFunctions')
BdvOptions = scyjava.jimport('bdv.util.BdvOptions')
VolatileTypeMatcher = scyjava.jimport('bdv.util.volatiles.VolatileTypeMatcher')
VolatileViews       = scyjava.jimport('bdv.util.volatiles.VolatileViews')

# BdvHandleFrame   = scyjava.jimport('bdv.util.BdvHandleFrame')
BdvPythonUtils   = scyjava.jimport('bdv.util.BdvPythonUtils')
RandomAccessibleIntervalSource   = scyjava.jimport('bdv.util.RandomAccessibleIntervalSource')
RandomAccessibleIntervalSource4D   = scyjava.jimport('bdv.util.RandomAccessibleIntervalSource4D')
AxisOrder   = scyjava.jimport('bdv.util.AxisOrder')

BasicViewerState = scyjava.jimport('bdv.viewer.BasicViewerState')
AffineTransform3D = scyjava.jimport('net.imglib2.realtransform.AffineTransform3D')
CacheControl = scyjava.jimport('bdv.cache.CacheControl')

MultiResolutionRenderer = scyjava.jimport('bdv.viewer.render.MultiResolutionRenderer')
PainterThread = scyjava.jimport('bdv.viewer.render.PainterThread')

BufferedImageOverlayRenderer = scyjava.jimport('bdv.viewer.render.awt.BufferedImageOverlayRenderer')

RequestRepaint = scyjava.jimport('bdv.viewer.RequestRepaint')

VolatileProjector = scyjava.jimport('bdv.viewer.render.VolatileProjector')
ScreenScales = scyjava.jimport('bdv.viewer.render.ScreenScales')

ProjectorFactory = scyjava.jimport('bdv.viewer.render.ProjectorFactory')
TiledProjector = scyjava.jimport('bdv.viewer.render.TiledProjector')
Tiling = scyjava.jimport('bdv.viewer.render.Tiling')
RenderStorage = scyjava.jimport('bdv.viewer.render.RenderStorage')
PythonRenderStorage = scyjava.jimport('bdv.viewer.render.PythonRenderStorage')
VisibleSourcesOnScreenBounds = scyjava.jimport('bdv.viewer.render.VisibleSourcesOnScreenBounds')

Callable = scyjava.jimport('java.util.concurrent.Callable')
ForkJoinPool = scyjava.jimport('java.util.concurrent.ForkJoinPool')
ForkJoinTask = scyjava.jimport('java.util.concurrent.ForkJoinTask')

File = scyjava.jimport('java.io.File')
ImageIO = scyjava.jimport('javax.imageio.ImageIO')

UnsignedByteType = scyjava.jimport('net.imglib2.type.numeric.integer.UnsignedByteType')

# ------ Begin the actual interesting stuff

path = '/Users/kharrington/Data/CREMI/sample_A_padded_20160501.hdf'
width = 800
height = 600

# ----- File loading

file       = h5py.File(path, 'r')
ds         = file['volumes/raw']
block_size = (32,) * 3
img, _     = imglyb.as_cell_img(ds, block_size, access_type='array', cache=10000)
try:
    # vimg is { RandomAccessibleInterval, }
    vimg   = VolatileViews.wrapAsVolatile(img)
except Exception as e:
    print(scyjava.jstacktrace(e))
    raise e

# ----- Create ImgLib2/BDV Data structures

# stackSource = BdvFunctions.show(vimg, 'test')
# Factor out BdvFunctions.show call

# stackSource = BdvPythonUtils.getStackSource(vimg, 'test')
# end show()

# Creating the SourceAndConverter

options = BdvOptions()

is_2D = True
axisOrder = AxisOrder.getAxisOrder(options.values.axisOrder(), img, is_2D)

sourceTransform = options.values.getSourceTransform()
name = 'BDVSourceDemo'

sources = ArrayList()
converterSetups = ArrayList()

_current_id = 0


def get_unused_id():
    global _current_id
    _current_id += 1
    return (_current_id - 1)


renderState = BasicViewerState()

stacks = AxisOrder.splitInputStackIntoSourceStacks(img, axisOrder)
for stack_id in range(stacks.size()):
    stack = stacks.get(stack_id)
    pixel_type = UnsignedByteType()
    if (stack.numDimensions() > 3):
        numTimepoints = stack.max(3) + 1
        source = RandomAccessibleIntervalSource4D((RandomAccessibleInterval) @ stack, pixel_type, sourceTransform, JString(name))
    else:
        source = RandomAccessibleIntervalSource(stack, pixel_type, sourceTransform, JString(name))
    source_type = source.getType()
    soc = BigDataViewer.wrapWithTransformedSource(
        SourceAndConverter(source, BigDataViewer.createConverterToARGB(pixel_type)))
    converterSetups.add(BigDataViewer.createConverterSetup(soc, get_unused_id()))
    sources.add(soc)
    renderState.addSource(soc)
    # bdv_stack_source = BdvStackSource()



# viewer = stackSource.getBdvHandle().getViewerPanel()


# Python note: this viewer state is roughly equivalent to napari's Dims


# renderState = BasicViewerState(viewer.state().snapshot())

# display = viewer.getDisplay()

# canvasW = viewer.getDisplay().getWidth()
# canvasH = viewer.getDisplay().getHeight()
canvasW = 800
canvasH = 600

affine = AffineTransform3D()
renderState.getViewerTransform(affine)

affine.set(affine.get(0, 3) - canvasW / 2, 0, 3)
affine.set(affine.get(1, 3) - canvasH / 2, 1, 3)
affine.scale(float(width) / canvasW)
affine.set(affine.get(0, 3) + width / 2, 0, 3)
affine.set(affine.get(1, 3) + height / 2, 1, 3)
renderState.setViewerTransform(affine)

# Prefs.showScaleBarInMovie()
scalebar = None

target = BufferedImageOverlayRenderer()

# We just need a dummy class because we aren't painting to a UI
@JImplements(RequestRepaint)
class RequestRepainter:
    @JOverride
    def requestRepaint():
        pass

    
screen_scale_factors = [1.0]
target_render_nanos = 0
num_rendering_threads = 1
# accumulate_projector = viewer.getOptionValues().getAccumulateProjectorFactory()
accumulate_projector = options.getAccumulateProjectorFactory()
use_volatile_if_available = False

renderer = MultiResolutionRenderer(target, RequestRepainter(), screen_scale_factors, target_render_nanos, num_rendering_threads, None, use_volatile_if_available, accumulate_projector, CacheControl.Dummy())

t = 0
renderState.setCurrentTimepoint(t)
# renderer.requestRepaint()
# renderer.paint(renderState)

# Trying to unpack renderer.paint

screenScales = ScreenScales(screen_scale_factors, target_render_nanos)
requestedScreenScaleIndex = 0
screenScale = screenScales.get(requestedScreenScaleIndex)

# Create memory for rendering
intervalResult = target.createRenderResult()
renderResult = target.getReusableRenderResult()
renderResult.init(width, height)# TODO check here
renderResult.setScaleFactor(screenScale.scale())

renderState.getViewerTransform(renderResult.getViewerTransform())
#
executors = ForkJoinPool(num_rendering_threads)

projectorFactory = ProjectorFactory(num_rendering_threads, executors, use_volatile_if_available, accumulate_projector)

# --- need to make a projector, this will be used for rendering
# projector = createProjector(renderState, requestedScreenScaleIndex, renderResult.getTargetImage(), 0, 0)

viewerState = renderState
screenImage = renderResult.getTargetImage()
screenTransform = viewerState.getViewerTransform().preConcatenate(screenScale.scaleTransform())
# screenTransform.translate( -offsetX, -offsetY, 0 )

onScreenBounds = VisibleSourcesOnScreenBounds(viewerState, screenImage, screenTransform)

tiles = Tiling.findTiles(onScreenBounds)
render_tiles = Tiling.splitForRendering(tiles)

numTiles = render_tiles.size()
tileProjectors = jpype.java.util.ArrayList()

for tile_id in range(numTiles):
    tile = render_tiles.get(tile_id)
    w = tile.tileSizeX()
    h = tile.tileSizeY()
    ox = tile.tileMinX()
    oy = tile.tileMinY()
    sources = tile.sources()

    # TODO finish creating the shared memory
    render_size = w * h
    num_visible_sources = sources.size()
    
    render_mask_ndarray = np.zeros((num_visible_sources, render_size),
                                   dtype=np.dtype('b'))
    render_image_ndarray = np.zeros((num_visible_sources, render_size), dtype=np.dtype(int))

    render_mask_jarray = JArray.of(render_mask_ndarray)
    render_image_jarray = JArray.of(render_image_ndarray)
    
    # render_mask_arrays = byte[ numVisibleSources ][ render_size ]
    # render_image_arrays = int[ numVisibleSources ][ render_size ]
    
    # Create a nested byte array
    # Create a nested int array
    tile_render_storage = PythonRenderStorage(render_mask_jarray, render_image_jarray)
    
    # tile_render_storage = PythonRenderStorage(w, h, sources.size())
    # tile_render_storage = RenderStorage(w, h, sources.size())

    tile_image = Views.interval(screenImage, Intervals.createMinSize(ox, oy, w, h))

    tile_projector = projectorFactory.createProjector(viewerState, sources, tile_image, screenTransform, tile_render_storage)
    
    tileProjectors.add(tile_projector)

projector = TiledProjector(tileProjectors)

# -- end of createProjector

requestNewFrameIfIncomplete = projectorFactory.requestNewFrameIfIncomplete()

# For multithreaded rendering
# trigger rendering
@JImplements(Callable)
class CreateProjector:
    @JOverride
    def call(self):
        projector.map(createProjector)
    
# success = executors.invoke(ForkJoinTask.adapt(CreateProjector()))
# currentScreenScaleIndex = requestedScreenScaleIndex


# Same thread
projector.map(False)

renderResult.setUpdated()
target.setRenderResult(renderResult)

bi = renderResult.getBufferedImage()

def buffered_image_to_ndarray(bi):
    array = np.zeros((bi.getWidth(), bi.getHeight()))
    
    for x in range(bi.getWidth()):
        for y in range(bi.getHeight()):
            color = bi.getData().getDataBuffer().getElem(x + y * bi.getWidth())
            array[x,y] = color# & 255

    return array

ImageIO.write(bi, "png", File("/Users/kharrington/Desktop/bdvlayer_002.png"))
array = buffered_image_to_ndarray(bi)

import napari

napari.view_image(array)

napari.run()


#bi.getRGB(0, 0)

# TODO
# Convert bi.getData().getDataBuffer() to a numpy array
# show in napari, then start a refresh loop


# TODO resume here

# scratch below

# bi = target.getReusableRenderResult().getBufferedImage()

# -----
# def create_bdv():
#     return None

# # Make a BigDataViewer instance
# bdv = create_bdv(options)

# def add_image(bdv, img):
#     pass

# # Add the image to BDV
# add_image(bdv, vimg)

# def bdv_to_image(bdv):
#     return None

# # Get an image from BDV and show it in matplotlib
# import matplotlib.pyplot as plt

# plt.imshow(bdv_to_image(bdv))


# bdv = BdvFunctions.show(vimg, 'raw')
# def runUntilBdvDoesNotShow():
#     panel = bdv.getBdvHandle().getViewerPanel()
#     while panel.isShowing():
#         time.sleep(0.3)
# threading.Thread(target=runUntilBdvDoesNotShow).start()
