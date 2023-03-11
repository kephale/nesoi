# From Lorenzo, https://raw.githubusercontent.com/brisvag/vispy/examples/instanced_mesh/examples/scene/instanced_mesh.py
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Instanced rendering of arbitrarily transformed meshes
=====================================================
"""

from vispy import app, gloo, visuals, scene, use
import numpy as np
from scipy.spatial.transform import Rotation
from vispy.io import read_mesh, load_data_file

import imageio.v3 as iio

# full gl+ context is required for instanced rendering
use(gl='gl+')


vertex_shader = """
// these attributes will be defined on an instance basis
attribute vec3 shift;
attribute vec4 color;
attribute vec3 transform_x;
attribute vec3 transform_y;
attribute vec3 transform_z;

varying vec4 v_color;

void main() {
    v_color = color;
    mat3 instance_transform = mat3(transform_x, transform_y, transform_z);
    vec3 pos_rotated = instance_transform * $position;
    vec4 pos_shifted = vec4(pos_rotated + shift, 1);
    gl_Position = $transform(pos_shifted);
}
"""

fragment_shader = """
varying vec4 v_color;

void main() {
  gl_FragColor = v_color;
}
"""


class OrientedMultiMeshVisual(visuals.Visual):
    def __init__(self, vertices, faces, positions, colors, transforms, subdivisions=5):
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)

        self.set_gl_state('translucent', depth_test=True, cull_face=True)
        self._draw_mode = 'triangles'

        # set up vertex and index buffer
        self.vbo = gloo.VertexBuffer(vertices.astype(np.float32))
        self.shared_program.vert['position'] = self.vbo
        self._index_buffer = gloo.IndexBuffer(data=faces.astype(np.uint32))

        # create a vertex buffer with a divisor argument of 1. This means that the
        # attribute value is set to the next element of the array every 1 instance.
        # The length of the array multiplied by the divisor determines the number
        # of instances
        self.shifts = gloo.VertexBuffer(positions.astype(np.float32), divisor=1)
        self.shared_program['shift'] = self.shifts

        # vispy does not handle matrix attributes (likely requires some big changes in GLIR)
        # so we decompose it into three vec3
        transforms = transforms.astype(np.float32)
        self.transforms_x = gloo.VertexBuffer(transforms[..., 0], divisor=1)
        self.transforms_y = gloo.VertexBuffer(transforms[..., 1], divisor=1)
        self.transforms_z = gloo.VertexBuffer(transforms[..., 2], divisor=1)
        self.shared_program['transform_x'] = self.transforms_x
        self.shared_program['transform_y'] = self.transforms_y
        self.shared_program['transform_z'] = self.transforms_z

        # we can provide additional buffers with different divisors, as long as the
        # amount of instances (length * divisor) is the same. In this case, we will change
        # color every 5 instances
        self.color = gloo.VertexBuffer(colors.astype(np.float32), divisor=1)
        self.shared_program['color'] = self.color

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.get_transform()

    def update_positions(self, new_positions):
        self.shifts = gloo.VertexBuffer(new_positions.astype(np.float32), divisor=1)
        self.shared_program['shift'] = self.shifts

    def update_colors(self, new_colors):
        self.color = gloo.VertexBuffer(new_colors.astype(np.float32), divisor=1)
        self.shared_program['color'] = self.color


# create a visual node class to add it to the canvas
OrientedMultiMesh = scene.visuals.create_visual_node(OrientedMultiMeshVisual)

# set up vanvas
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'arcball'
view.camera.scale_factor = 1000

N = 10000

width = 1000
height = 1000

edt_filename = "/Users/kharrington/nesoi/resources/napari_edt.png"
edt = iio.imread(edt_filename)

colormap_filename = "/Users/kharrington/nesoi/resources/napari_logo_binary_centeronly.png"
colormap = iio.imread(colormap_filename)

mesh_file = load_data_file('orig/triceratops.obj.gz')
vertices, faces, _, _ = read_mesh(mesh_file)

np.random.seed(0)
pos = (np.random.rand(N, 3) - 0.5) * width
colors = np.random.rand(N, 4)
transforms = Rotation.random(N).as_matrix()

multimesh = OrientedMultiMesh(vertices * 10, faces, pos, colors, transforms, parent=view.scene)
# global transforms are applied correctly after the individual instance transforms!
multimesh.transform = visuals.transforms.STTransform(scale=(3, 2, 1))

import time
from napari.qt.threading import thread_worker

def get_path_weight(pt):
    x = int(pt[0] + width / 2)
    y = int(pt[1] + width / 2)

    # Out of bounds
    if x < 0 or x >= width or y < 0 or y >= width:
        return 255
    
    return edt[x, y]

PURPLE = (93/255.0, 83/255.0, 100/255.0, 1)
CYAN = (118/255.0, 175/255.0, 175/255.0, 1)

def get_color(pt):
    x = int(pt[0] + width / 2)
    y = int(pt[1] + width / 2)

    # Out of bounds
    if x < 0 or x >= width or y < 0 or y >= width:
        return PURPLE
    elif colormap[x,y] > 0:
        return CYAN
    else:
        return PURPLE
    

@thread_worker
def animator(initial_pos, initial_color):
    this_pos = initial_pos
    this_color = initial_color
    for t in range(10000):        
        
        for idx in range(N):
            w = get_path_weight(this_pos[idx,:])
            if w < 5:
                w = 0.01
            else:
                w /= 5

            # Jiggle a bit
            this_pos[idx,:] += w * (np.random.rand(1, 3) - 0.5).squeeze()
            if this_pos[idx,0] < -width / 2:
                this_pos[idx,0] = -width / 2
            elif this_pos[idx,0] > width / 2:
                this_pos[idx,0] = width / 2

            if this_pos[idx,1] < -width / 2:
                this_pos[idx,1] = -width / 2
            elif this_pos[idx,1] > width / 2:
                this_pos[idx,1] = width / 2

            if this_pos[idx,2] < -width / 2:
                this_pos[idx,2] = -width / 2
            elif this_pos[idx,2] > width /2:
                this_pos[idx,2] = width / 2

            # Set colors
            this_color[idx,:] = get_color(this_pos[idx,:])

        multimesh.update_colors(this_color)
        multimesh.update_positions(this_pos)
        time.sleep(0.02)
        print(f"Update {t}")
        multimesh.update()

anim = animator(pos, colors)
anim.start()
        
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()
