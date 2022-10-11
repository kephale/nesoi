import numpy as np
import pandas as pd

from tyssue import Sheet, SheetGeometry
from tyssue.draw.vispy_draw import sheet_view
from tyssue.generation import three_faces_sheet

# This is from https://github.com/DamCB/tyssue/blob/d5da9a47fcf9f027ed9daabc3fa6cccaaa3a6d86/tests/draw/test_vispy.py

def view():
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
    canvas, view = sheet_view(sheet, face=face_spec, edge=edge_spec, interactive=True)
    content = view.scene.children
    edge_mesh, face_mesh = content[-2:]

view()
