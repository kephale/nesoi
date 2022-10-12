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
from tyssue.draw.ipv_draw import _get_meshes, _wire_color_from_sequence, _face_color_from_sequence
from tyssue.io.hdf5 import load_datasets
from tyssue.utils.utils import get_sub_eptm
# from .tyssue_utils import view_sheet

print('tyssue version', tyssue.__version__)

# Loading datasets

# Read pre-recorded datasets
h5store = 'data/small_hexagonal.hf5'
from tyssue.io.hdf5 import save_datasets, load_datasets

import pickle

history = pickle.load(open( "data/apoptosis_history.pickle", "rb" ))

# Data prep

def edge_mesh(sheet, coords, **edge_specs):
    """
    Creates a ipyvolume Mesh of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    spec = sheet_spec()["edge"]
    spec.update(**edge_specs)
    if callable(spec["color"]):
        spec["color"] = spec["color"](sheet)

    if isinstance(spec["color"], str):
        color = spec["color"]
    elif hasattr(spec["color"], "__len__"):
        color = _wire_color_from_sequence(spec, sheet)[:, :3]

    u, v, w = coords
    mesh = (np.hcat((sheet.vert_df[u], sheet.vert_df[v], sheet.vert_df[w])),
            sheet.edge_df[["srce", "trgt"]].astype(dtype=np.uint32),
            color,)
    return mesh


def face_mesh(sheet, coords, **face_draw_specs):
    """
    Creates a ipyvolume Mesh of the face polygons
    """
    Ne, Nf = sheet.Ne, sheet.Nf
    if callable(face_draw_specs["color"]):
        face_draw_specs["color"] = face_draw_specs["color"](sheet)

    if isinstance(face_draw_specs["color"], str):
        color = face_draw_specs["color"]

    elif hasattr(face_draw_specs["color"], "__len__"):
        color = _face_color_from_sequence(face_draw_specs, sheet)[:, :3]

    if "visible" in sheet.face_df.columns:
        edges = sheet.edge_df[sheet.upcast_face(sheet.face_df["visible"])].index
        _sheet = get_sub_eptm(sheet, edges)
        if _sheet is not None:
            sheet = _sheet
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                edges = edges.values.astype(np.uint32)
                indexer = np.concatenate([faces, edges + Nf, edges + Ne + Nf])
                color = color.take(indexer, axis=0)

    epsilon = face_draw_specs.get("epsilon", 0)
    up_srce = sheet.edge_df[["s" + c for c in coords]]
    up_trgt = sheet.edge_df[["t" + c for c in coords]]

    Ne, Nf = sheet.Ne, sheet.Nf

    if epsilon > 0:
        up_face = sheet.edge_df[["f" + c for c in coords]].values
        up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
        up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    mesh_ = np.concatenate(
        [sheet.face_df[coords].values, up_srce.values, up_trgt.values]
    )

    triangles = np.vstack(
        [sheet.edge_df["face"], np.arange(Ne) + Nf, np.arange(Ne) + Ne + Nf]
    ).T.astype(dtype=np.uint32)

    color = np.linspace(0, 1, len(mesh_))
    
    mesh = (mesh_ * 10.0, triangles, color)
    return mesh


# Get a mesh timeseries
def _get_meshes(sheet, coords, draw_specs):

    meshes = []
    edge_spec = draw_specs["edge"]
    edge_spec["visible"] = False
    if edge_spec["visible"]:
        edges = edge_mesh(sheet, coords, **edge_spec)
        meshes.append(edges)
    else:
        edges = None

    face_spec = draw_specs["face"]
    face_spec["visible"] = True
    if face_spec["visible"]:
        faces = face_mesh(sheet, coords, **face_spec)
        meshes.append(faces)
    else:
        faces = None        
        
    print('faces', faces)
    return meshes


def browse_history(
    history, coords=["x", "y", "z"], start=None, stop=None, size=None, **draw_specs_kw
):
    times = history.slice(start, stop, size)
    num_frames = times.size
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
    sheet = history.retrieve(0)
#    ipv.clear()
    fig, meshes = view_sheet(sheet, coords, **draw_specs_kw)

    lim_inf = sheet.vert_df[sheet.coords].min().min()
    lim_sup = sheet.vert_df[sheet.coords].max().max()
    # ipv.xyzlim(lim_inf, lim_sup)

    def set_frame(i=0):
        fig.animation = 0
        t = times[i]
        meshes = _get_meshes(history.retrieve(t), coords, draw_specs)
        update_view(fig, meshes)

    # ipv.show()
    
    # interact(set_frame, i=(0, num_frames - 1))
    
# browse_history(history, edge={"color":lambda s : s.edge_df["length"]})
# browse_history(history)

specs_kw = {}
draw_specs = sheet_spec()
spec_updater(draw_specs, specs_kw)
coords = ["x", "y", "z"]

sheet = history.retrieve(0)
meshes = _get_meshes(sheet, coords, draw_specs)
mesh = meshes[0]
print(f"mesh: ({mesh[0].shape}, {mesh[1].shape}, {mesh[2].shape})")

# napari stuff

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

surface = viewer.add_surface(mesh, colormap='turbo', opacity=0.9, contrast_limits=[0, 1], name='tyssue')

def updater():
    import time

    print('starting to sleep for 10')
    time.sleep(10)
    print('done sleeping')
    time.sleep(1)
    
    for t in range(len(history)):
        print(f"Time {t}")
        sheet = history.retrieve(t)
        meshes = _get_meshes(sheet, coords, draw_specs)
        mesh = meshes[0]
        print(f"mesh: ({mesh[0].shape}, {mesh[1].shape}, {mesh[2].shape})")

        surface.data = mesh
        time.sleep(0.5)

from threading import Thread

thread = Thread(target=updater)
thread.start()
        
# fig, mesh = view_sheet(sheet, coords=['z', 'x', 'y'], mode="3D", interactive=True)
# fig
