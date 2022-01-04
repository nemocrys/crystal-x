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

# DOLFINx Imports
import dolfinx

from geometry.geometry import create_geometry
from geometry.geometry import Volume, Interface, Surface, Boundary
from geometry.gmsh_helpers import gmsh_model_to_mesh

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

