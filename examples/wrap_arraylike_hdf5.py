import h5py
import numpy as np
import imglyb
import scyjava
import threading
import time
import jpype
from jpype import JImplements, JOverride

scyjava.config.add_repositories({'jitpack.io': 'https://jitpack.io'})

scyjava.config.endpoints.append('org.scijava:ui-behaviour:2.0.7')
scyjava.config.endpoints.append('com.github.kephale:bigdataviewer-core:2640f84d13')
scyjava.config.endpoints.append('sc.fiji:bigdataviewer-vistools:1.0.0-beta-32-SNAPSHOT')
scyjava.start_jvm()

path = '/Users/kharrington/Data/CREMI/sample_A_padded_20160501.hdf'
width = 800
height = 600

Views = scyjava.jimport('net.imglib2.view.Views')
Intervals = scyjava.jimport('net.imglib2.util.Intervals')

BdvFunctions        = scyjava.jimport('bdv.util.BdvFunctions')
BdvOptions = scyjava.jimport('bdv.util.BdvOptions')
VolatileTypeMatcher = scyjava.jimport('bdv.util.volatiles.VolatileTypeMatcher')
VolatileViews       = scyjava.jimport('bdv.util.volatiles.VolatileViews')

BdvHandleFrame   = scyjava.jimport('bdv.util.BdvHandleFrame')

BasicViewerState = scyjava.jimport('bdv.viewer.BasicViewerState')
AffineTransform3D = scyjava.jimport('net.imglib2.realtransform.AffineTransform3D')
CacheControl = scyjava.jimport('bdv.cache.CacheControl')

MultiResolutionRenderer = scyjava.jimport('bdv.viewer.render.MultiResolutionRenderer')
PainterThread = scyjava.jimport('bdv.viewer.render.PainterThread')

BufferedImageOverlayRenderer = scyjava.jimport('bdv.viewer.render.awt.BufferedImageOverlayRenderer')

RequestRepaint = scyjava.jimport('bdv.viewer.RequestRepaint')

file       = h5py.File(path, 'r')
ds         = file['volumes/raw']
block_size = (32,) * 3
img, _     = imglyb.as_cell_img(ds, block_size, access_type='array', cache=10000)
try:
    vimg   = VolatileViews.wrapAsVolatile(img)
except Exception as e:
    print(scyjava.jstacktrace(e))
    raise e

stackSource = BdvFunctions.show(vimg, 'test')

viewer = stackSource.getBdvHandle().getViewerPanel()
renderState = BasicViewerState(viewer.state().snapshot())

display = viewer.getDisplay()

canvasW = viewer.getDisplay().getWidth()
canvasH = viewer.getDisplay().getHeight()

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

# @JImplements
# class RequestRepainter:
#    def getReusableRenderResult(self):
#        pass

#    def createRenderResult(self):
#        return BufferedImageRenderResult()

#    def setRenderResult(self, renderResult):
#        pass

#    def getWidth():
       
# 				return width;
# 			}

# 			@Override
# 			public int getHeight()
# 			{
# 				return height;
# 			}   



@JImplements(RequestRepaint)
class RequestRepainter:
    @JOverride
    def requestRepaint():
        pass

screen_scale_factors = [1.0]
target_render_nanos = 0
num_rendering_threads = 1
accumulate_projector = viewer.getOptionValues().getAccumulateProjectorFactory()
use_volatile_if_available = False

renderer = MultiResolutionRenderer(target, RequestRepainter(), screen_scale_factors, target_render_nanos, num_rendering_threads, None, use_volatile_if_available, accumulate_projector, CacheControl.Dummy())

t = 0
renderState.setCurrentTimepoint(t)
# renderer.requestRepaint()
#renderer.paint(renderState)

# Trying to unpack renderer.paint

VolatileProjector = scyjava.jimport('bdv.viewer.render.VolatileProjector')
ScreenScales = scyjava.jimport('bdv.viewer.render.ScreenScales')

ProjectorFactory = scyjava.jimport('bdv.viewer.render.ProjectorFactory')
TiledProjector = scyjava.jimport('bdv.viewer.render.TiledProjector')
Tiling = scyjava.jimport('bdv.viewer.render.Tiling')
RenderStorage = scyjava.jimport('bdv.viewer.render.RenderStorage')
VisibleSourcesOnScreenBounds = scyjava.jimport('bdv.viewer.render.VisibleSourcesOnScreenBounds')

Callable = scyjava.jimport('java.util.concurrent.Callable')
ForkJoinPool = scyjava.jimport('java.util.concurrent.ForkJoinPool')
ForkJoinTask = scyjava.jimport('java.util.concurrent.ForkJoinTask')

screenScales = ScreenScales(screen_scale_factors, target_render_nanos)
requestedScreenScaleIndex = 0
screenScale = screenScales.get(requestedScreenScaleIndex)

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

for tile in range(numTiles):
    tile = render_tiles.get(tile)
    w = tile.tileSizeX()
    h = tile.tileSizeY()
    ox = tile.tileMinX()
    oy = tile.tileMinY()
    sources = tile.sources()

    tile_render_storage = RenderStorage(w, h, sources.size())

    tile_image = Views.interval(screenImage, Intervals.createMinSize(ox, oy, w, h))
    tileProjectors.add(tile)

projector = TiledProjector(tileProjectors)

# -- end of createProjector

requestNewFrameIfIncomplete = projectorFactory.requestNewFrameIfIncomplete()

# trigger rendering
@JImplements(Callable)
class CreateProjector:
    @JOverride
    def call():
        projector.map(createProjector)
    
success = executors.invoke(ForkJoinTask.adapt(CreateProjector()))

currentScreenScaleIndex = requestedScreenScaleIndex

renderResult.setUpdated()
target.setRenderResult(renderResult)

# scratch below

bi = target.getReusableRenderResult().getBufferedImage()

# -----
def create_bdv():
    return None

# Make a BigDataViewer instance
bdv = create_bdv(options)

def add_image(bdv, img):
    pass

# Add the image to BDV
add_image(bdv, vimg)

def bdv_to_image(bdv):
    return None

# Get an image from BDV and show it in matplotlib
import matplotlib.pyplot as plt

plt.imshow(bdv_to_image(bdv))


# bdv = BdvFunctions.show(vimg, 'raw')
# def runUntilBdvDoesNotShow():
#     panel = bdv.getBdvHandle().getViewerPanel()
#     while panel.isShowing():
#         time.sleep(0.3)
# threading.Thread(target=runUntilBdvDoesNotShow).start()
