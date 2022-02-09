"""
Run with dolfinx using Docker:
docker run -it --mount type=bind,source="$(pwd)",target=/root dolfinx/dolfinx:latest

install additional stuff

pip3 install pyelmer
source /usr/local/bin/dolfinx-complex-mode

TODO create own Dockerfile for all this

by M. Schröder, A. Enders-Seidlitz, B. E. Abali, K. Dadzis
"""


#####################################################################################################
#                                                                                                   #
#                                          IMPORTS                                                  #
#                                                                                                   #
#####################################################################################################

# General Imports
import numpy as np
import yaml
from time import time

#---------------------------------------------------------------------------------------------------#

# DOLFINx Imports
import dolfinx
from geometry.czochralski import crystal
import ufl
from petsc4py import PETSc
from mpi4py import MPI

#---------------------------------------------------------------------------------------------------#

# Geometry Imports
from geometry.geometry import create_geometry
from geometry.geometry import Volume, Interface, Surface, Boundary
from geometry.gmsh_helpers import gmsh_model_to_mesh

#---------------------------------------------------------------------------------------------------#

# crystal-x Imports
from crystalx.equations.maxwell import Maxwell
from crystalx.equations.heat import Heat
from crystalx.equations.laplace import Laplace
from crystalx.equations.interface import Stefan
from crystalx.time_stepper import OneStepTheta
from crystalx.transient.auxiliary_methods import interface_normal, normal_velocity, interface_displacement, meniscus_displacement

#---------------------------------------------------------------------------------------------------#

# Check if complex mode is activated
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError(
        "Complex mode required. Activate it with 'source /usr/local/bin/dolfinx-complex-mode'."
    )

#---------------------------------------------------------------------------------------------------#

# Start timer
start_time = time()

#####################################################################################################
#                                                                                                   #
#                                          LOAD MESH                                                #
#                                                                                                   #
#####################################################################################################

gdim = 2  # gmsh dimension / geometric dimension
gmsh_model = create_geometry()

# Loading of mesh, cell tags and facet tags
# mesh, cell_tags, facet_tags = gmsh_model_to_mesh(
#     gmsh_model, cell_data=True, facet_data=True, gdim=gdim
# )

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="mesh")
    cell_tags = xdmf.read_meshtags(mesh, name="Cell tags")
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(mesh, name="Facet tags")
#####################################################################################################
#                                                                                                   #
#                                   SETTING FUNCTION SPACES                                         #
#                                                                                                   #
#####################################################################################################

# Normal Vector
N = ufl.FacetNormal(mesh)

#---------------------------------------------------------------------------------------------------#

# Volume Element
dV = ufl.Measure(
    "dx",
    domain=mesh,
    subdomain_data=cell_tags,  # cells_mat,
    metadata={"quadrature_degree": 2, "quadrature_scheme": "uflacs"},
)

# Boundary Element (for boundaries on the outside of the computational domain)
dA = ufl.Measure(
    "ds",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": 2, "quadrature_scheme": "uflacs"},
)

# Interface Element (for boundaries on the inside of the computational domain)
dI = ufl.Measure(
    "dS",
    domain=mesh,
    subdomain_data=facet_tags,
    metadata={"quadrature_degree": 2, "quadrature_scheme": "uflacs"},
)

#---------------------------------------------------------------------------------------------------#

# Finite Element basis (for scalar elements)
scalar_element = lambda degree: ufl.FiniteElement(
    "CG", mesh.ufl_cell(), degree
)


vector_element = lambda degree: ufl.VectorElement(
    "CG", mesh.ufl_cell(), degree
)

scalar_element_DG = lambda degree: ufl.FiniteElement(
    "DG", mesh.ufl_cell(), degree
)


vector_element_DG = lambda degree: ufl.VectorElement(
    "DG", mesh.ufl_cell(), degree
)
#---------------------------------------------------------------------------------------------------#

# Function Space for Maxwell Equation
Space_A = dolfinx.FunctionSpace(mesh, scalar_element(degree=1))  # A

# Function Space for Heat Equation
Space_T = dolfinx.FunctionSpace(mesh, scalar_element(degree=2))  # T

# Function Space for Interface Displacement in normal direction
Space_V = dolfinx.FunctionSpace(mesh, scalar_element(degree=1))  # V

# Function Space for Mesh Movement
Space_MM = dolfinx.FunctionSpace(mesh, vector_element(degree=1)) # MM
#####################################################################################################
#                                                                                                   #
#                                       PARAMETERS                                                  #
#                                                                                                   #
#####################################################################################################

# Ambient Temperature
T_amb = 293.15#300.0  # K

# Heat source
f_heat = 0

h = 5  # W / (m^2 K)

#---------------------------------------------------------------------------------------------------#

Dt = 1e-4
t_end = 10 * Dt

v_pull = 4  # mm/min
v_pull *= 1.6666666e-5  # m/s

#---------------------------------------------------------------------------------------------------#

# permittivity
eps_0 = 8.85e-12  # in A s/(V m)
# permeability
mu_0 = 1.25663706e-6  # in V s/(A m)

# frequency 
freq = 13.5e3  # in Hz
# current frequency
omega = 2 * np.pi * freq
current_density = 100 * 35367.76513153229  # current [A] / Area [m^2]

current = dolfinx.Constant(
    mesh, 0
) 

#####################################################################################################
#                                                                                                   #
#                                   MATERIAL COEFFICIENTS                                           #
#                                                                                                   #
#####################################################################################################

# Discontinuous function space for material coefficients
Q = dolfinx.FunctionSpace(mesh, ("DG", 0))

# heat conductivity
kappa = dolfinx.Function(Q, name="kappa")
# electric conductivity
varsigma = dolfinx.Function(Q, name="varsigma")
# emissivity
varepsilon = dolfinx.Function(Q, name="varepsilon")
# density
rho = dolfinx.Function(Q, name="rho")
# heat capacity
capacity = dolfinx.Function(Q, name="capacity")

with open("examples/materials/materials.yml") as f:
    material_data = yaml.safe_load(f)

#---------------------------------------------------------------------------------------------------#

with kappa.vector.localForm() as loc_kappa, varsigma.vector.localForm() as loc_varsigma, varepsilon.vector.localForm() as loc_varepsilon, rho.vector.localForm() as loc_rho, capacity.vector.localForm() as loc_capacity:
    for vol in Volume:
        cells = cell_tags.indices[cell_tags.values == vol.value]
        num_cells = len(cells)
        loc_kappa.setValues(
            cells, np.full(num_cells, material_data[vol.name]["Heat Conductivity"])
        )
        loc_varsigma.setValues(
            cells, np.full(num_cells, material_data[vol.name]["Electric Conductivity"])
        )
        loc_varepsilon.setValues(
            cells, np.full(num_cells, material_data[vol.name]["Emissivity"])
        )
        loc_rho.setValues(cells, np.full(num_cells, material_data[vol.name]["Density"]))
        loc_capacity.setValues(
            cells, np.full(num_cells, material_data[vol.name]["Heat Capacity"])
        )

#####################################################################################################
#                                                                                                   #
#                                 SOLVE MAXWELLS EQUATIONS                                          #
#                                                                                                   #
#####################################################################################################
sourrounding_facets = facet_tags.indices[
        facet_tags.values == Boundary.surrounding.value
    ]
dofs_A = dolfinx.fem.locate_dofs_topological(Space_A, 1, sourrounding_facets)

value_A = dolfinx.Function(Space_A)
with value_A.vector.localForm() as loc:  # according to https://jorgensd.github.io/dolfinx-tutorial/chapter2/ns_code2.html#boundary-conditions
    loc.set(0)
bcs_A = []# [dolfinx.DirichletBC(value_A, dofs_A)]

em_problem = Maxwell(Space_A)
em_form = em_problem.setup(em_problem.solution, dV, dA, dI, mu_0, omega, varsigma, current_density)
em_problem.assemble(em_form, bcs_A)
A = em_problem.solve()

#####################################################################################################
#                                                                                                   #
#                                   ASSEMBLE HEAT EQUATION                                          #
#                                                                                                   #
#####################################################################################################

sourrounding_facets = facet_tags.indices[
        facet_tags.values == Boundary.surrounding.value
    ]

insulation_bottom_facets = facet_tags.indices[
    facet_tags.values == Surface.insulation_bottom.value
]

coil_inside_facets = facet_tags.indices[
    facet_tags.values == Surface.inductor_inside.value
]

#---------------------------------------------------------------------------------------------------#

dofs_T = dolfinx.fem.locate_dofs_topological(
    Space_T, 1, np.concatenate([coil_inside_facets, sourrounding_facets, insulation_bottom_facets])
)

value_T = dolfinx.Function(Space_T)
with value_T.vector.localForm() as loc:
    loc.set(T_amb)
bcs_T = [dolfinx.DirichletBC(value_T, dofs_T)]

#---------------------------------------------------------------------------------------------------#

heat_problem = Heat(Space_T)

# Set initial temperature via stationary simulation
heat_form = heat_problem.setup(heat_problem.solution, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat)
heat_problem.assemble(heat_form, bcs_T)
_ = heat_problem.solve()

T_old = dolfinx.Function(Space_T)
with heat_problem.solution.vector.localForm() as loc_T, T_old.vector.localForm() as loc_T_old:
    # loc_T.set(T_amb)
    # loc_T_old.set(T_amb)
    loc_T.copy(loc_T_old)


one_step_theta_timestepper = OneStepTheta(heat_problem, 0.5+Dt)

# heat_form = heat_problem.setup(heat_problem.solution, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat)
heat_form = one_step_theta_timestepper.step(T_old, rho * capacity, 0.0, Dt, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat)

heat_problem.assemble(heat_form, bcs_T)

#####################################################################################################
#                                                                                                   #
#                                   ASSEMBLE STEFAN PROBLEM                                         #
#                                                                                                   #
#####################################################################################################

melt_crystal_interface_facets = facet_tags.indices[
        facet_tags.values == Interface.melt_crystal.value
    ]

dofs_melt_crystal_interface = dolfinx.fem.locate_dofs_topological(
    Space_V, 1, melt_crystal_interface_facets 
)

bcs_V = []

with open("examples/materials/materials.yml") as f:
    mat_data = yaml.safe_load(f)

latent_heat_value = 5.96e4 * mat_data["melt"]["Density"] # J/m^3

# #---------------------------------------------------------------------------------------------------#

stefan_problem = Stefan(Space_V)
stefan_a, stefan_L = stefan_problem.setup(stefan_problem.solution, dV, dA, dI, kappa, latent_heat_value, heat_problem.solution)
stefan_problem.assemble(stefan_a, stefan_L, bcs_V, dofs_melt_crystal_interface)
_ , normal_velocity_values = stefan_problem.solve(dofs_melt_crystal_interface)

# #---------------------------------------------------------------------------------------------------#

n = interface_normal(Space_V, Interface.melt_crystal, facet_tags)
normal_velocity_vector = np.repeat(normal_velocity_values.reshape(-1,1), n.shape[1], axis=1) * n 

#####################################################################################################
#                                                                                                   #
#                                   ASSEMBLE MESH MOVEMENT                                          #
#                                                                                                   #
#####################################################################################################

sourrounding_facets = facet_tags.indices[
        facet_tags.values == Boundary.surrounding.value
    ]

symmetry_axis_facets = facet_tags.indices[
        facet_tags.values == Boundary.symmetry_axis.value
    ]

crystal_surface_facets = facet_tags.indices[
        facet_tags.values == Surface.crystal.value
    ]

crucible_surface_facets = facet_tags.indices[
        facet_tags.values == Surface.crucible.value
    ]

melt_crystal_interface_facets = facet_tags.indices[
        facet_tags.values == Interface.melt_crystal.value
    ]

#---------------------------------------------------------------------------------------------------#

dofs_hom_dirichlet = dolfinx.fem.locate_dofs_topological(
    Space_MM, 1, np.concatenate([sourrounding_facets, crucible_surface_facets, crystal_surface_facets]) # TODO: DoFs on crystal surface not quite right 
)

value_MM = dolfinx.Function(Space_MM)
with value_MM.vector.localForm() as loc:
    loc.set(0)
bcs_MM = [dolfinx.DirichletBC(value_MM, dofs_hom_dirichlet)]

#---------------------------------------------------------------------------------------------------#

dofs_symmetry_axis = dolfinx.fem.locate_dofs_topological(
    (Space_MM.sub(0), Space_MM.sub(0).collapse(),),
    1,
    symmetry_axis_facets,
)

value_MM = dolfinx.Function(Space_MM.sub(0).collapse()) # only BC on x-component
with value_MM.vector.localForm() as loc:
    loc.set(0)

bcs_MM.append(dolfinx.DirichletBC(
        value_MM, dofs_symmetry_axis, Space_MM.sub(0)
    )
)

#---------------------------------------------------------------------------------------------------#
# Calculate Displacement Vector on Interface as u = Dt * ((n * (v_pull + v_growth)) n)

v_pull_vector = v_pull * np.repeat(np.array([0.0 , 1.0, 0.0]).reshape(3,1), normal_velocity_vector.shape[0] , axis=1).T

velocity_vector = normal_velocity_vector + v_pull_vector

normal_velocity_vector = normal_velocity(velocity_vector, n)

beta = 10.0 # in degrees
beta *= np.pi / 180

displacement_vector = interface_displacement(normal_velocity_vector, v_pull_vector, Dt, beta, Space_V, Interface.melt_crystal, Surface.melt, facet_tags)
#---------------------------------------------------------------------------------------------------#

dofs_melt_crystal_interface = dolfinx.fem.locate_dofs_topological(
    Space_MM, 1, melt_crystal_interface_facets 
)

displacement_function = dolfinx.Function(Space_MM, name="displacement_function")

with displacement_function.vector.localForm() as loc:
    values = loc.getArray()
    values[2 * dofs_melt_crystal_interface] = displacement_vector[:,0]
    values[2 * dofs_melt_crystal_interface + 1] = displacement_vector[:,1]
    loc.setArray(values)

#---------------------------------------------------------------------------------------------------#
# Calculate the displacement caused by the change of the meniscus shape 
# meniscus_displacement(displacement_function, Space_MM, Surface.melt, facet_tags) # TODO: Remove error on calculate meniscus

#####################################################################################################
#                                                                                                   #
#                                         TIME LOOP                                                 #
#                                                                                                   #
#####################################################################################################

res_dir = "examples/results/"
vtk = dolfinx.io.VTKFile(MPI.COMM_WORLD, res_dir + "result.pvd", "w")

for step, t in enumerate(np.arange(0.0, t_end + Dt, Dt)):
    
    fields = [heat_problem.solution, stefan_problem.solution, displacement_function]
    output_fields = [sol._cpp_object for sol in fields]
    vtk.write_function(output_fields, t)


    heat_problem.solve()
    # mesh_movement_problem.solve()

    with heat_problem.solution.vector.localForm() as loc_T, T_old.vector.localForm() as loc_T_old:
        loc_T.copy(loc_T_old)

    if MPI.COMM_WORLD.rank == 0:
        elapsed = int(time() - start_time)
        e_h, e_m, e_s = (
            int(elapsed / 3600),
            int(elapsed % 3600 / 60),
            int((elapsed % 3600) % 60),
        )
        print(
            f"step {step}: time {t:.8f} s --- temperature solution in {e_h:.0f} h {e_m:.0f} min {e_s:.0f} s "
        )

vtk.close()



# #####################################################################################################
# #                                                                                                   #
# #                                          OUTPUT                                                   #
# #                                                                                                   #
# #####################################################################################################

# res_dir = "examples/results/"

# fields = [A, T]
# output_fields = [sol._cpp_object for sol in fields]

# vtk = dolfinx.io.VTKFile(MPI.COMM_WORLD, res_dir + "result.pvd", "w")

# vtk.write_function(output_fields)
# vtk.close()