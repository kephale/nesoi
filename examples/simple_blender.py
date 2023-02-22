import bpy
import shutil

blender_bin = "/Applications/Blender.app/Contents/MacOS/Blender"
if blender_bin:
   print("Found:", blender_bin)
   bpy.app.binary_path = blender_bin
else:
   print("Unable to find blender!")

bpy.ops.mesh.primitive_cone_add(location=(-3, 0, 0))
output = '/Users/kharrington/Desktop/render.png'
bpy.context.scene.render.filepath = output
bpy.ops.render.render(write_still=True)

