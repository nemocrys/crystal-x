import yaml
import gmsh
import numpy as np
from .czochralski import crucible, surrounding
from objectgmsh import Model, Shape, factory, MeshControlLinear, MeshControlExponential
# this is a package I created für Elmer simulations
from .czochralski import crucible, melt, crystal, inductor, seed, crucible_adapter ,crucible_support, axis_top, surrounding
import os

# this was copied from another project


# load geometry parameter

# create Enum for diffrent volumes, interfaces, surfaces and boundaries
from enum import Enum

# volumes (dim = 2)
class Volume(Enum):
    
    def __new__(cls, *args, **kwds):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, value, material): 
        self.material = material

    crucible = 1, "graphite-CZ3R6300"
    melt = 2, "tin-liquid"
    crystal = 3, "tin-solid"
    inductor = 4, "copper-inductor"
    seed = 5, "tin-solid"
    insulation = 6, "insulation"
    adapter = 7, "graphite-CZ3R6300"
    axis_bottom = 8, "steel-1.4541"
    axis_top = 9, "steel-1.4541"
    surrounding = 10, "air"
    surrounding_in_inductor = 11, "air"

# interfaces between solid/solid or solid/liquid volumes (dim = 1)
class Interface(Enum):
    crucible_melt = 12
    melt_crystal = 13
    crucible_insulation = 14

# interface between solid/gas or liquid/gas volumes (dim = 1)
class Surface(Enum):
    crystal = 15
    meniscus = 16
    melt_flat = 17
    crucible = 18
    seed = 19
    insulation = 20
    adapter = 21
    axis_bottom = 22
    axis_top = 23
    inductor = 24

# boundaries of the domain (dim = 1)
class Boundary(Enum):
    symmetry_axis = 25
    surrounding = 26
    axis_bottom = 27
    axis_top = 28
    inductor_inside = 29

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
    Seed = seed(model, 2, **config["seed"], crystal=Crystal)
    Ins = crucible_support(
        model, 2, **config["insulation"], top_shape=Crucible, name="insulation"
    )
    Adp = crucible_adapter(model, 2, **config["crucible_adapter"], top_shape=Ins)
    Ax_bt = crucible_support(
        model, 2, **config["axis_bt"], top_shape=Adp, name="axis_bt"
    )
    Ax_top = axis_top(model, 2, **config["axis_top"], seed=Seed, l=config["surrounding"]["X0"][1] + config["surrounding"]["h"] - Seed.params.X0[1] - Seed.params.l)

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
    
    melt_surf_ids = Melt.get_interface(Surrounding)
    x_min_of_surf_melt = [factory.get_bounding_box(1, melt_surf_ids[0])[0], factory.get_bounding_box(1, melt_surf_ids[1])[0]]
    surf_meniscus = Shape(model, 1, "surf_meniscus", [melt_surf_ids[np.argmin(x_min_of_surf_melt)]])
    surf_melt_flat = Shape(model, 1, "surf_melt_flat", [melt_surf_ids[np.argmax(x_min_of_surf_melt)]])
    
    surf_crucible = Shape(
        model, 1, "surf_crucible", Crucible.get_interface(Surrounding)
    )
    surf_seed = Shape(model, 1, "surf_seed", Seed.get_interface(Surrounding))
    surf_ins = Shape(model, 1, "surf_ins", Ins.get_interface(Surrounding))
    surf_adp = Shape(model, 1, "surf_adp", Adp.get_interface(Surrounding))
    surf_axbt = Shape(model, 1, "surf_axbt", Ax_bt.get_interface(Surrounding))
    surf_axtop = Shape(model, 1, "surf_axtop", Ax_top.get_interface(Surrounding))

    surf_inductor = Shape(model, 1, "surf_inductor", Inductor.get_interface(Surrounding))

    # symmetry axis
    sym_ax = Shape(model, 1, "sym_ax", model.symmetry_axis)

    # boundaries
    bnd_surrounding = Shape(
        model,
        1,
        "bnd_surrounding",
        [x for x in Surrounding.boundaries if x not in model.symmetry_axis + surf_axtop.geo_ids + surf_seed.geo_ids + surf_crystal.geo_ids + surf_meniscus.geo_ids + surf_melt_flat.geo_ids + surf_crucible.geo_ids + surf_ins.geo_ids + surf_adp.geo_ids + surf_axbt.geo_ids + surf_inductor.geo_ids]
    )

    bnd_axbt = Shape(model, 1, "bnd_axbt", [Ax_bt.bottom_boundary])
    bnd_axtop = Shape(model, 1, "bnd_axtop", [Ax_top.top_boundary])

    bnd_inductor_inside = Shape(
        model, 1, "bnd_inductor_inside", Inductor.get_interface(Surrounding_in_inductor)
    )
    model.make_physical()

    # meshing
    model.deactivate_characteristic_length()
    model.set_const_mesh_sizes()  # these are defined with geometry.py
    # set linear constraints for mesh size transition
    for shape in [Melt, Crystal, Seed, Ax_top, Crucible, Ins, Adp, Ax_bt]:
        MeshControlLinear(model, shape, shape.mesh_size, Surrounding.mesh_size)
    # exponential constraints for refinement
    MeshControlExponential(
        model, if_melt_crystal, Crystal.params.r / 50, exp=1.6, fact=3
    )
    MeshControlExponential(model, surf_meniscus, Melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(model, surf_melt_flat, Melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(model, if_crucible_melt, Melt.mesh_size / 5, exp=1.6, fact=3)
    MeshControlExponential(
        model, surf_crucible, Crucible.mesh_size / 3, exp=1.6, fact=3
    )
    MeshControlExponential(model, Inductor, Inductor.mesh_size)

    model.generate_mesh(
        size_factor=1
    )  # increase / decrease size factor to adapt number of cells
    # model.show()  # uncomment this to visualize the mesh
    model.write_msh("./mesh.msh2")
    if os.path.exists("./mesh.msh"):
        os.remove("./mesh.msh")
    os.rename("./mesh.msh2", "./mesh.msh")

    print("2D shapes:")
    for shape in model.get_shapes(2):
        print(shape.name, shape.ph_id)
    print()
    print("1D shapes:")
    for shape in model.get_shapes(1):
        print(shape.name, shape.ph_id)

    return gmsh.model


if __name__ == "__main__":
    create_geometry()
