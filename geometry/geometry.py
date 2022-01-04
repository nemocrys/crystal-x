import yaml
from .czochralski import crucible, surrounding
from pyelmer.gmsh import *

# this is a package I created für Elmer simulations
from .czochralski import crucible, melt, crystal, inductor, crucible_support, surrounding
import os

# this was copied from another project


# load geometry parameter

# create Enum for diffrent volumes, interfaces, surfaces and boundaries
from enum import Enum

# volumes (dim = 2)
class Volume(Enum):
    crucible = 1
    melt = 2
    crystal = 3
    inductor = 4
    insulation = 5
    surrounding = 6
    surrounding_in_inductor = 7

# interfaces between solid/solid or solid/liquid volumes (dim = 1)
class Interface(Enum):
    crucible_melt = 8
    melt_crystal = 9
    crucible_insulation = 10

# interface between solid/gas or liquid/gas volumes (dim = 1)
class Surface(Enum):
    crystal = 11
    melt = 12
    crucible = 13
    insulation = 14
    inductor = 15
    insulation_bottom = 18
    inductor_inside = 19

# boundaries of the domain (dim = 1)
class Boundary(Enum):
    symmetry_axis = 16
    surrounding = 17

# create geometry
def create_geometry():
    model = Model()

    with open("geometry/config_geo.yml") as f:
        config = yaml.safe_load(f)

    Crucible = crucible(model, 2, **config["crucible"])
    Melt = melt(
        model, 2, Crucible, **config["melt"], crystal_radius=config["crystal"]["r"],
    )
    Crystal = crystal(model, 2, **config["crystal"], melt=Melt)
    Inductor = inductor(model, 2, **config["inductor"])
    Ins = crucible_support(
        model, 2, **config["insulation"], top_shape=Crucible, name="insulation"
    )
    Surrounding = surrounding(model, 2, **config["surrounding"])
    Surrounding_in_inductor = Shape(
        model,
        2,
        "surrounding_in_inductor",
        Surrounding.get_part_in_box(
            [
                Inductor.params.X0[0] - Inductor.params.d_in / 2,
                Inductor.params.X0[0] + Inductor.params.d_in / 2,
            ],
            [Inductor.params.X0[1] - Inductor.params.d_in, 1e6],
        ),
    )
    Surrounding -= Surrounding_in_inductor
    model.synchronize()

    # interfaces between bodies
    if_crucible_melt = Shape(model, 1, "if_crucible_melt", Crucible.get_interface(Melt))
    if_melt_crystal = Shape(model, 1, "if_melt_crystal", Melt.get_interface(Crystal))
    if_crucible_ins = Shape(model, 1, "if_crucible_ins", Crucible.get_interface(Ins))

    # surfaces for radiation / convective cooling
    surf_crystal = Shape(model, 1, "surf_crystal", Crystal.get_interface(Surrounding))
    surf_melt = Shape(model, 1, "surf_melt", Melt.get_interface(Surrounding))
    surf_crucible = Shape(
        model, 1, "surf_crucible", Crucible.get_interface(Surrounding)
    )
    surf_ins = Shape(model, 1, "surf_ins", [Ins.left_boundary, Ins.right_boundary])
    surf_inductor = Shape(model, 1, "suf_inductor", Inductor.get_interface(Surrounding))

    # symmetry axis
    sym_ax = Shape(model, 1, "sym_ax", model.symmetry_axis)

    # boundaries
    bnd_surrounding = Shape(
        model,
        1,
        "bnd_surrounding",
        [
            Surrounding.top_boundary,
            Surrounding.right_boundary,
            Surrounding.bottom_boundary,
        ],
    )
    bnd_ins_bottom = Shape(model, 1, "bnd_ins_bottom", [Ins.bottom_boundary])
    bnd_inductor_inside = Shape(
        model, 1, "bnd_inductor_inside", Inductor.get_interface(Surrounding_in_inductor)
    )
    model.make_physical()

    # meshing
    model.deactivate_characteristic_length()
    model.set_const_mesh_sizes()  # these are defined with geometry.py
    # set linear constraints for mesh size transition
    for shape in [Melt, Crystal, Crucible, Ins, Inductor]:
        MeshControlLinear(model, shape, shape.mesh_size, Surrounding.mesh_size)
    # exponential constraints for refinement
    MeshControlExponential(
        model, if_melt_crystal, Crystal.params.r / 30, exp=1.6, fact=3
    )
    MeshControlExponential(model, surf_melt, Melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(model, if_crucible_melt, Melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(
        model, surf_crucible, Crucible.mesh_size / 3, exp=1.6, fact=3
    )
    MeshControlExponential(model, Inductor, Inductor.mesh_size)

    model.generate_mesh(
        size_factor=2
    )  # increase / decrease size factor to adapt number of cells
    # model.show()  # uncomment this to visualize the mesh
    model.write_msh("./mesh.msh2")
    if os.path.exists("./mesh.msh"):
        os.remove("./mesh.msh")
    os.rename("./mesh.msh2", "./mesh.msh")

    return gmsh.model


if __name__ == "__main__":
    create_geometry()
