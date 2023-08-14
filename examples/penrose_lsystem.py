import napari
import numpy as np

# based on https://rosettacode.org/wiki/Penrose_tiling

def drawpenrose(num_steps=4):
    lindenmayer_rules = {
        "A": "",
        "M": "OA++PA----NA[-OA----MA]++", 
        "N": "+OA--PA[---MA--NA]+",
        "O": "-MA++NA[+++OA++PA]-", 
        "P": "--OA++++MA[+PA++++NA]--NA"
    }

    def rul(x):
        return lindenmayer_rules.get(x, x)

    penrose = "[N]++[N]++[N]++[N]++[N]"
    for _ in range(num_steps):
        penrose = ''.join([rul(x) for x in penrose])

    x, y, theta, r = 160, 160, np.pi / 5, 20.0
    lines, stack = [], []

    for c in penrose:
        if c == "A":
            xx, yy = x + r * np.cos(theta), y + r * np.sin(theta)
            lines.append([[x, y], [xx, yy]])
            x, y = xx, yy
        elif c == "+":
            theta += np.pi / 5
        elif c == "-":
            theta -= np.pi / 5
        elif c == "[":
            stack.append([x, y, theta])
        elif c == "]":
            x, y, theta = stack.pop()

    viewer = napari.Viewer()
    viewer.add_shapes(lines, shape_type='line', edge_color='orange', edge_width=2)

    napari.run()

drawpenrose(num_steps=6)
