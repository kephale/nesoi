import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt

import ipywidgets as widgets
from IPython.display import display, Image

import napari
import vispy as vp

import tyssue
from tyssue import Sheet, History
from tyssue import config

from tyssue import SheetGeometry as geom
from tyssue.dynamics.sheet_vertex_model import SheetModel as basemodel
from tyssue.dynamics.apoptosis_model import SheetApoptosisModel as model
from tyssue.solvers.quasistatic import QSSolver

from tyssue.config.draw import sheet_spec
from tyssue.utils.utils import spec_updater

from tyssue.draw.vispy_draw import sheet_view, face_visual, edge_visual
# from tyssue.draw import sheet_view, create_gif, browse_history

from tyssue.io.hdf5 import save_datasets, load_datasets

import pickle

def view_sheet(sheet, coords=None, interactive=True, viewer=None, **draw_specs_kw):
    """Uses VisPy to display an epithelium"""
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    if viewer is None:
        viewer = napari.Viewer()

    canvas = viewer.window.qt_viewer.canvas
    
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    view.camera.aspect = 1
    view.bgcolor = vp.color.Color("#222222")
    if draw_specs["edge"]["visible"]:
        wire = edge_visual(sheet, coords, **draw_specs["edge"])
        view.add(wire)
    if draw_specs["face"]["visible"]:
        mesh = face_visual(sheet, coords, **draw_specs["face"])
        view.add(mesh)

    canvas.show()
    view.camera.set_range()
    if interactive:
        napari.run()
    return canvas, view

