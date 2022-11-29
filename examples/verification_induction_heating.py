"""
Use the dockerfile provided in the repository 
Run with dolfinx using Docker:
docker run -it --rm -v ${PWD}:/home/workdir nemocrys/dolfinx:v0.5.2 bash 

source /usr/local/bin/dolfinx-complex-mode

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
import os

#---------------------------------------------------------------------------------------------------#

# DOLFINx Imports
import dolfinx
import ufl
from petsc4py import PETSc
from mpi4py import MPI


# crystal-x Equations
from crystalx.steadystate.equations.maxwell import Maxwell

# geometry modeling
from objectgmsh import Model, Shape, MeshControlExponential
import gmsh
from dolfinx.io.gmshio import model_to_mesh

#---------------------------------------------------------------------------------------------------#

# Check if complex mode is activated
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    raise RuntimeError(
        "Complex mode required. Activate it with 'source /usr/local/bin/dolfinx-complex-mode'."
    )

#---------------------------------------------------------------------------------------------------#

# setup working directory

with open("examples/setup_verification_induction_heating.yml") as f:
    config = yaml.safe_load(f)
wdir = f"results_air-size={config['air_size_factor']*(config['coil_d']+config['coil_r_i'])}_mesh-size={config['mesh_size_factor']}_winding-d={config['coil_d']}"
if not os.path.exists(wdir):
    os.mkdir(wdir)

#####################################################################################################
#                                                                                                   #
#                                          GENERATE MESH                                                #
#                                                                                                   #
#####################################################################################################


model = Model()
occ = gmsh.model.occ

# parameters
air_size = config["air_size_factor"] * (config["coil_r_i"] + config["coil_d"])

# geometry modeling
cylinder = Shape(model, 2, "cylinder", [occ.addRectangle(0, 0, 0, config["graphite_r"], config["height"])])
coil = Shape(model, 2, "coil", [occ.addRectangle(config["coil_r_i"], 0, 0, config["coil_d"], config["height"])])
air = Shape(model, 2, "air", [occ.addRectangle(0, 0, 0, air_size, config["height"])])
air.geo_ids = [x[1] for x in occ.cut(air.dimtags, cylinder.dimtags + coil.dimtags, removeTool=False)[0]]
occ.synchronize()
# boundaries
cylinder_surf = Shape(model, 1, "cylinder_surf", [cylinder.right_boundary])
outside_surf = Shape(model, 1, "outside_surf", [air.right_boundary])

model.make_physical()

# mesh
mesh_size_base = config["height"] * 0.2
mesh_size_min = config["height"] * 0.01
cylinder.mesh_size = mesh_size_base
air.mesh_size = mesh_size_base
coil.mesh_size = mesh_size_base
MeshControlExponential(model, coil, mesh_size_min)

model.deactivate_characteristic_length()
model.set_const_mesh_sizes()
model.generate_mesh(size_factor=config["mesh_size_factor"])
model.write_msh(f"{wdir}/mesh.msh")

# Loading of mesh, cell tags and facet tags
gdim = 2  # gmsh dimension / geometric dimension
model_rank = 0
mesh, cell_tags, facet_tags = model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
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

#---------------------------------------------------------------------------------------------------#

# Function Space for Maxwell Equation
Space_A = dolfinx.fem.FunctionSpace(mesh, scalar_element(degree=1))  # A

#####################################################################################################
#                                                                                                   #
#                                       PARAMETERS                                                  #
#                                                                                                   #
#####################################################################################################

# frequency 
freq = config["frequency"]  # in Hz
omega = 2 * np.pi * freq
current = config["current"]
current_density = current / (config["height"] * config["coil_d"])

#####################################################################################################
#                                                                                                   #
#                                   MATERIAL COEFFICIENTS                                           #
#                                                                                                   #
#####################################################################################################

# Discontinuous function space for material coefficients
Q = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))

# electric conductivity
varsigma = dolfinx.fem.Function(Q, name="varsigma")

# permeability
mu = dolfinx.fem.Function(Q, name="mu")

#---------------------------------------------------------------------------------------------------#

with varsigma.vector.localForm() as loc_varsigma, mu.vector.localForm() as loc_mu:
    for vol in [coil, air]:
        cells = cell_tags.find(vol.ph_id)
        loc_varsigma.setValues(
            cells, np.full(len(cells), 0)
        )
    cells = cell_tags.find(cylinder.ph_id)
    loc_varsigma.setValues(
        cells, np.full(len(cells), config["varsigma_graphite"])
    )
    loc_mu.set(config["mu_0"])

#####################################################################################################
#                                                                                                   #
#                                 SOLVE MAXWELLS EQUATIONS                                          #
#                                                                                                   #
#####################################################################################################

outside_facets = facet_tags.indices[
        facet_tags.values == outside_surf.ph_id
    ]
dofs_A = dolfinx.fem.locate_dofs_topological(Space_A, 1, outside_facets)

value_A = dolfinx.fem.Function(Space_A)
with value_A.vector.localForm() as loc:
    loc.set(0)
bcs_A = [dolfinx.fem.dirichletbc(value_A, dofs_A)]

em_problem = Maxwell(Space_A)
em_form = em_problem.setup(em_problem.solution, dV, dA, dI, mu, omega, varsigma, current_density, coil.ph_id)
em_problem.assemble(em_form, bcs_A)
em_problem.solve()

# vtk = dolfinx.io.VTKFile(MPI.COMM_WORLD, f"{wdir}/result.pvd", "w")
# vtk.write_function(em_problem.solution, 0)
# vtk.write_function(varsigma, 0)
# vtk.close()
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{wdir}/solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(em_problem.solution)
    xdmf.write_function(varsigma)

