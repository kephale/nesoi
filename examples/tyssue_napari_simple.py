import numpy as np
import pandas as pd

import napari
import vispy as vp

from tyssue import Sheet, SheetGeometry
from tyssue.config.draw import sheet_spec
from tyssue.utils.utils import spec_updater
from tyssue.draw.vispy_draw import sheet_view, face_visual, edge_visual
from tyssue.generation import three_faces_sheet

# This is a napari version of https://github.com/DamCB/tyssue/blob/d5da9a47fcf9f027ed9daabc3fa6cccaaa3a6d86/tests/draw/test_vispy.py

def sheet_napari(sheet, coords=None, interactive=True, **draw_specs_kw):
    """Uses VisPy to display an epithelium"""
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    viewer = napari.Viewer()

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

def view_napari():
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    face_spec = {
        "color": pd.Series(range(3)),
        "color_range": (0, 3),
        "visible": True,
        "colormap": "Blues",
        "epsilon": 0.1,
    }

    color = pd.DataFrame(
        np.zeros((sheet.Ne, 3)), index=sheet.edge_df.index, columns=["R", "G", "B"]
    )

    color.loc[0, "R"] = 0.8

    edge_spec = {"color": color, "visible": True}
    canvas, view = sheet_napari(sheet, face=face_spec, edge=edge_spec, interactive=True)
    content = view.scene.children
    edge_mesh, face_mesh = content[-2:]

# This will show tyssue in napari
view_napari()
