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
import ufl
from petsc4py import PETSc
from mpi4py import MPI

#---------------------------------------------------------------------------------------------------#

# Geometry Imports
from geometry.geometry import create_geometry
from geometry.geometry import Volume, Interface, Surface, Boundary
from geometry.gmsh_helpers import gmsh_model_to_mesh

#---------------------------------------------------------------------------------------------------#
# steady state Imports

# crystal-x Equations
from crystalx.steadystate.equations.maxwell import Maxwell
from crystalx.steadystate.equations.heat import Heat

# crystal-x auxiliary methods
from crystalx.steadystate.auxiliary_methods import set_temperature_scaling, mesh_move, interface_displacement

#---------------------------------------------------------------------------------------------------#

# crystal-x helpers
from crystalx.helpers import save_function, save_mesh

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
mesh, cell_tags, facet_tags = gmsh_model_to_mesh(
    gmsh_model, cell_data=True, facet_data=True, gdim=gdim
)

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
Space_T = dolfinx.FunctionSpace(mesh, scalar_element(degree=1))  # T

# Function Space for Interface Displacement in normal direction
Space_V = dolfinx.FunctionSpace(mesh, scalar_element(degree=1))  # V

# Function Space for Mesh Movement
Space_MM = dolfinx.FunctionSpace(mesh, vector_element(degree=1)) # MM
#####################################################################################################
#                                                                                                   #
#                                       PARAMETERS                                                  #
#                                                                                                   #
#####################################################################################################
with open("examples/materials/materials.yml") as f:
    material_data = yaml.safe_load(f)

# Ambient Temperature
T_amb = 293.15#300.0  # K

# Melting Temperature
T_melt = material_data["tin-solid"]["Melting Point"] # 505.08K

# Heat source
f_heat = 0

h = 5  # W / (m^2 K)

#---------------------------------------------------------------------------------------------------#

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
            cells, np.full(num_cells, material_data[vol.material]["Heat Conductivity"])
        )
        loc_varsigma.setValues(
            cells, np.full(num_cells, material_data[vol.material]["Electric Conductivity"])
        )
        loc_varepsilon.setValues(
            cells, np.full(num_cells, material_data[vol.material]["Emissivity"])
        )
        loc_rho.setValues(cells, np.full(num_cells, material_data[vol.material]["Density"]))
        loc_capacity.setValues(
            cells, np.full(num_cells, material_data[vol.material]["Heat Capacity"])
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

axis_bottom_facets = facet_tags.indices[
        facet_tags.values == Boundary.axis_bottom.value
    ]

axis_top_facets = facet_tags.indices[
        facet_tags.values == Boundary.axis_top.value
    ]

coil_inside_facets = facet_tags.indices[
    facet_tags.values == Boundary.inductor_inside.value
]

#---------------------------------------------------------------------------------------------------#

dofs_T = dolfinx.fem.locate_dofs_topological(
    Space_T, 1, np.concatenate([axis_bottom_facets, axis_top_facets, coil_inside_facets, sourrounding_facets])
)

value_T = dolfinx.Function(Space_T)
with value_T.vector.localForm() as loc:
    loc.set(T_amb)
bcs_T = [dolfinx.DirichletBC(value_T, dofs_T)]

#---------------------------------------------------------------------------------------------------#

heat_problem = Heat(Space_T)
 

#####################################################################################################
#                                                                                                   #
#                                      SOLVING LOOP                                                 #
#                                                                                                   #
#####################################################################################################

res_dir = "examples/results/"
vtk = dolfinx.io.VTKFile(MPI.COMM_WORLD, res_dir + "steady_state_result.pvd", "w")

for iteration in range(6):
    print(f"Mesh update iteration {iteration}")
    set_temperature_scaling(heat_problem, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat, bcs_T, desired_temp=T_melt, interface=Interface.melt_crystal, facet_tags=facet_tags)
    heat_form = heat_problem.setup(heat_problem.solution, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat)
    heat_problem.assemble(heat_form, bcs_T)

    _ = heat_problem.solve()

    displacement_function = interface_displacement(heat_problem.solution, T_melt, Volume, Boundary, Surface, Interface, cell_tags, facet_tags)

    fields = [heat_problem.solution, displacement_function]
    output_fields = [sol._cpp_object for sol in fields]
    vtk.write_function(output_fields, iteration)
    mesh_move(mesh, displacement_function)

vtk.close()

print(heat_problem._heat_scaling)

# #####################################################################################################
# #                                                                                                   #
# #                                          OUTPUT                                                   #
# #                                                                                                   #
# #####################################################################################################

res_dir = "examples/results/"

# Save Mesh as initial geometry for transient
save_mesh(mesh, cell_tags, facet_tags, MPI.COMM_WORLD)

# Save steadystate solution as initial condition
save_function(heat_problem.solution, name="steadystate_solution")
