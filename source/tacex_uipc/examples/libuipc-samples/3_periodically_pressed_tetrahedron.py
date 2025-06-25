import numpy as np

import uipc
from uipc import Logger, Timer
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, flip_inward_triangles
from uipc.constitution import AffineBodyConstitution
from uipc.gui import SceneGUI 
from uipc.unit import MPa, GPa
from asset_dir import AssetDir

Timer.enable_all()
Logger.set_level(Logger.Level.Warn)

workspace = AssetDir.output_path(__file__)
engine = Engine('cuda', workspace)
world = World(engine)
config = Scene.default_config()
dt = 0.02
config['dt'] = dt
config['gravity'] = [[0.0], [-9.8], [0.0]]
scene = Scene(config)

# create constitution and contact model
abd = AffineBodyConstitution()

# friction ratio and contact resistance
scene.contact_tabular().default_model(0.5, 1.0 * GPa)
default_element = scene.contact_tabular().default_element()

# create a regular tetrahedron
Vs = np.array([[0,1,0],
               [0,0,1],
               [-np.sqrt(3)/2, 0, -0.5],
               [np.sqrt(3)/2, 0, -0.5]])
Ts = np.array([[0,1,2,3]])

# setup a base mesh to reduce the later work
base_mesh = tetmesh(Vs, Ts)
# apply the constitution and contact model to the base mesh
abd.apply_to(base_mesh, 100 * MPa)
# apply the default contact model to the base mesh
default_element.apply_to(base_mesh)

# label the surface, enable the contact
label_surface(base_mesh)
# label the triangle orientation to export the correct surface mesh
label_triangle_orient(base_mesh)
# flip the triangles inward for better rendering
base_mesh = flip_inward_triangles(base_mesh)

mesh1 = base_mesh.copy()
pos_view = uipc.view(mesh1.positions())
# move the mesh up for 1 unit
pos_view += uipc.Vector3.UnitY() * 1.5

mesh2 = base_mesh.copy()
is_fixed = mesh2.instances().find(uipc.builtin.is_fixed)
is_fixed_view = uipc.view(is_fixed)
is_fixed_view[:] = 1

# create objects
object1 = scene.objects().create("upper_tet")
object1.geometries().create(mesh1)

object2 = scene.objects().create("lower_tet")
object2.geometries().create(mesh2)

world.init(scene)
sgui = SceneGUI(scene)


run = False
def on_update():
    global run
    if(imgui.Button('run & stop')):
        run = not run

    if(run):
        world.advance()
        world.retrieve()
        sgui.update()

ps.set_user_callback(on_update)
ps.show()