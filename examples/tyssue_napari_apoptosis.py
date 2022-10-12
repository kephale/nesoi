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
from tyssue.io.hdf5 import load_datasets

print('tyssue version', tyssue.__version__)

# Loading datasets

# Read pre-recorded datasets
h5store = 'data/small_hexagonal.hf5'
from tyssue.io.hdf5 import save_datasets, load_datasets

datasets = load_datasets(h5store,
                         data_names=['face', 'vert', 'edge'])
# Corresponding specifications
specs = config.geometry.cylindrical_sheet()
sheet = Sheet('emin', datasets, specs)
sheet.sanitize(trim_borders=True, order_edges=True)

geom.update_all(sheet)
# Model
nondim_specs = config.dynamics.quasistatic_sheet_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs)

sheet.get_opposite()
live_edges = sheet.edge_df[sheet.edge_df['opposite']==-1].index
dead_src = sheet.edge_df.loc[live_edges, 'srce'].unique()

### Boundary conditions
sheet.vert_df.is_active = 1
sheet.vert_df.loc[dead_src, 'is_active'] = 0

sheet.edge_df['is_active'] = sheet.upcast_srce('is_active') * sheet.upcast_trgt('is_active')

# Energy minimization

min_settings = {
#    "minimize":{
        'options': {
            'disp': False,
            'ftol': 1e-6,
            'gtol': 1e-5},
#    }
}
solver = QSSolver()

res = solver.find_energy_min(sheet, geom, model, **min_settings)
print(res['success'])

# Custom display function

def leg_joint_view(sheet, coords=['z', 'x', 'y']):
    
    geom.update_all(sheet)
    x, y, z = coords
    datasets = {}
    
    datasets['face'] = sheet.face_df.sort_values(z)
    datasets['vert'] = sheet.vert_df.sort_values(z)
    edge_z = 0.5 * (sheet.upcast_srce(sheet.vert_df[z]) +
                    sheet.upcast_trgt(sheet.vert_df[z]))
    datasets['edge'] = sheet.edge_df.copy()
    datasets['edge'][z] = edge_z
    datasets['edge'] = datasets['edge'].sort_values(z)
    
    tmp_sheet = Sheet('tmp', datasets,
                      sheet.specs)
    tmp_sheet.reset_index()

    draw_specs = {
        'vert': {
            'visible': False
            },
        'edge': {
            'color': tmp_sheet.edge_df[z],
            #'zorder': depth.values
            }
        }
    
    fig, ax = sheet_view(tmp_sheet, coords[:2], mode='2D', **draw_specs)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-10, 10)
    ax.set_facecolor('#404040')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_size_inches((10, 12))
    return fig, ax

# fig, ax = leg_joint_view(sheet)

# Choose apoptotic cell

apoptotic_cell = 16
print('Apoptotic cell position:\n{}'.format(sheet.face_df.loc[apoptotic_cell, sheet.coords]))
apoptotic_edges = sheet.edge_df[sheet.edge_df['face'] == apoptotic_cell]
apoptotic_verts = apoptotic_edges['srce'].values
print("Indices of the apoptotic vertices: {}".format(apoptotic_verts))

from tyssue.behaviors.sheet import apoptosis
from tyssue.behaviors import EventManager


manager = EventManager('face')


sheet.settings['apoptosis'] = {
    'shrink_rate': 1.2,
    'critical_area': 8.,
    'radial_tension': 0.2,
    'contractile_increase': 0.3,
    'contract_span': 2
    }

sheet.face_df['id'] = sheet.face_df.index.values
manager.append(apoptosis, face_id=apoptotic_cell, **sheet.settings['apoptosis'])

# Run the simulation

t = 0
stop=100

progress = widgets.IntProgress(min=0, max=stop)
progress.value = 0
display(progress)

history = History(sheet)

while manager.current and t < stop:
    manager.execute(sheet)
    t += 1
    progress.value = t
    res = solver.find_energy_min(sheet, geom, model, **min_settings)
    history.record()
    manager.update()


color = sheet.vert_df['y']


def sheet_napari(sheet, coords=None, interactive=True, **draw_specs_kw):
    """Uses VisPy to display an epithelium"""
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    viewer = napari.Viewer()
    # napari.view_image(np.zeros((10, 10, 10)))

    # Vispy scene creation
    # if coords is None:
    #     coords = ["x", "y", "z"]
    # canvas = scene.SceneCanvas(keys="interactive", show=True, size=(1240, 720))

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


# fig, mesh = sheet_napari(sheet, coords=['z', 'x', 'y'], edge={"color":color}, mode="3D", interactive=True)
fig, mesh = sheet_napari(sheet, coords=['z', 'x', 'y'], mode="3D", interactive=True)
fig
