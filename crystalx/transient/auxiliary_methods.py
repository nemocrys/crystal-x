import numpy as np
from scipy.linalg import solve
from petsc4py import PETSc
import dolfinx
from geometry.geometry import Interface
import ufl
from mpi4py import MPI

from .equations.laplace import Laplace

def reset_values(function):
    with function.vector.localForm() as loc:
        loc.set(0)

def interface_normal(function_space, interface, facet_tags):
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        function_space, 1, interface_facets
    )

    #---------------------------------------------------------------------------------------------------#
    
    coordinates_interface = function_space.tabulate_dof_coordinates()[dofs_interface]

    # permutation in ascending order (x-Coordinate)
    permutation_interface = np.argsort(coordinates_interface[:, 0])
    
    inverse_permutation = np.empty(permutation_interface.size, dtype=np.int32)
    for i in np.arange(permutation_interface.size):
        inverse_permutation[permutation_interface[i]] = i

    coordinates_interface = coordinates_interface[permutation_interface]

    # calculate connection vectors
    connection_vectors = coordinates_interface[1:,:] - coordinates_interface[:-1,:]
    
    # calculate tanget vector
    orthogonal_vectors = np.zeros(connection_vectors.shape)
    orthogonal_vectors[:,0] = connection_vectors[:,1]
    orthogonal_vectors[:,1] -= connection_vectors[:,0]
    # normalize
    orthogonal_vectors /= np.repeat(np.linalg.norm(orthogonal_vectors, axis = 1).reshape(-1,1), 3, axis=1)

    # take the mean of the two adjacent facets for the normal on a vertex
    normals = np.zeros(shape=coordinates_interface.shape)
    
    for i in range(len(orthogonal_vectors) - 1):
        normals[i+1,:2] = 0.5 * (orthogonal_vectors[i,:2] + orthogonal_vectors[i+1, :2])
    
    # the normal on the symm. axis is -e_2
    normals[0,1] = -1.0

    #---------------------------------------------------------------------------------------------------#
    # the normal of the triple point is the normal of its facet
    
    normals[-1,:] = orthogonal_vectors[-1,:]

    #---------------------------------------------------------------------------------------------------#
    normals = normals[inverse_permutation]
    
    return normals

# def normal_velocity(velocity_vector, function_space, interface, meniscus, crystal, facet_tags):
#     interface_facets = facet_tags.indices[
#         facet_tags.values == interface.value
#     ]

#     dofs_interface = dolfinx.fem.locate_dofs_topological(
#         function_space, 1, interface_facets
#     )

#     #---------------------------------------------------------------------------------------------------#

#     meniscus_facets = facet_tags.indices[
#         facet_tags.values == meniscus.value
#     ]

#     dofs_meniscus = dolfinx.fem.locate_dofs_topological(
#         function_space, 1, meniscus_facets
#     )

#     #---------------------------------------------------------------------------------------------------#

#     crystal_facets = facet_tags.indices[
#         facet_tags.values == crystal.value
#     ]

#     dofs_crystal = dolfinx.fem.locate_dofs_topological(
#         function_space, 1, crystal_facets
#     )

#     #---------------------------------------------------------------------------------------------------#
    
#     coordinates_interface = function_space.tabulate_dof_coordinates()[dofs_interface]

#     # permutation in ascending order (x-Coordinate)
#     permutation_interface = np.argsort(coordinates_interface[:, 0])
    
#     inverse_permutation = np.empty(permutation_interface.size, dtype=np.int32)
#     for i in np.arange(permutation_interface.size):
#         inverse_permutation[permutation_interface[i]] = i

#     # permute coordinates and velocities in the same way
#     coordinates_interface = coordinates_interface[permutation_interface]

#     # calculate connection vectors
#     connection_vectors = coordinates_interface[1:,:] - coordinates_interface[:-1,:]
    
#     # calculate tanget vector
#     orthogonal_vectors = np.zeros(connection_vectors.shape)
#     orthogonal_vectors[:,0] = connection_vectors[:,1]
#     orthogonal_vectors[:,1] -= connection_vectors[:,0]
#     # normalize
#     orthogonal_vectors /= np.repeat(np.linalg.norm(orthogonal_vectors, axis = 1).reshape(-1,1), 3, axis=1)

#     # take the mean of the two adjacent facets for the normal on a vertex
#     normals = np.zeros(shape=coordinates_interface.shape)
    
#     for i in range(len(orthogonal_vectors) - 1):
#         normals[i+1,:2] = 0.5 * (orthogonal_vectors[i,:2] + orthogonal_vectors[i+1, :2])
    
#     # the normal on the symm. axis is -e_2
#     normals[0,1] = -1.0


#     #---------------------------------------------------------------------------------------------------#

#     # the normal on the triple point is the tangent to the meniscus or the tangent to the crystal.
#     # It depends on the angle theta between the resulting velocity at the triple point and the tangent to the meniscus
#     coordinates_meniscus = function_space.tabulate_dof_coordinates()[dofs_meniscus]

#     # permutation in decresing order (y-Coordinate)
#     permutation_meniscus = np.flipud(np.argsort(coordinates_meniscus[:, 1]))

#     coordinates_meniscus = coordinates_meniscus[permutation_meniscus]

#     # calculate tangent direction from first two points
#     tangent_meniscus = coordinates_meniscus[1,:] - coordinates_meniscus[0,:]
#     tangent_meniscus /= np.repeat(np.linalg.norm(tangent_meniscus), 3)

#     # angle theta between the resulting velocity at the triple point and the tangent to the meniscus
#     theta = np.arccos(np.dot(velocity_vector[-1, :], tangent_meniscus) / np.linalg.norm(velocity_vector[-1, :])) 

#     if theta <= np.pi:
#         normals[-1,:] = tangent_meniscus
#     else:
#         # Calculate the direction of the crystal
#         coordinates_crystal = function_space.tabulate_dof_coordinates()[dofs_crystal]
#         # permutation in ascending order (y-Coordinate)
#         permutation_crystal = np.argsort(coordinates_crystal[:, 1])
#         coordinates_crystal = coordinates_crystal[permutation_crystal]

#         # calculate tangent direction from first two points
#         tangent_crystal = coordinates_crystal[1,:] - coordinates_crystal[0,:]
#         tangent_crystal /= np.repeat(np.linalg.norm(tangent_crystal), 3)

#         normals[-1,:] = tangent_crystal

#     #---------------------------------------------------------------------------------------------------#
#     normals = normals[inverse_permutation]

#     # TODO: maybe a bit to many unnecessary computations
#     normal_projection = np.diag(normals @ velocity_vector.T).reshape(-1,1)
#     normal_velocity_vector = np.repeat(normal_projection, 3, axis = 1) * normals
    
#     return normal_velocity_vector
def mesh_displacement(displacement_function, Volume, Boundary, Surface, Interface, cell_tags, facet_tags):
    melt = Volume.melt
    crystal = Volume.crystal
    interface = Interface.melt_crystal
    meniscus = Surface.meniscus

    # Setup mesh movement problem
    mesh = displacement_function.function_space.mesh
    
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

    vector_element = ufl.VectorElement(
    "CG", mesh.ufl_cell(), 1
    )

    laplace_problem = Laplace(displacement_function.function_space)
    form_MM = laplace_problem.setup(laplace_problem.solution, dV, dA, dI)


    #---------------------------------------------------------------------------------------------------#
    # Set Dirichlet Boundary Conditions
    
    #---------------------------------------------------------------------------------------------------#
    # set displacement as dirichlet BC
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    meniscus_facets = facet_tags.indices[
        facet_tags.values == meniscus.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        displacement_function.function_space, 1, interface_facets
    )

    dofs_meniscus = dolfinx.fem.locate_dofs_topological(
        displacement_function.function_space, 1, meniscus_facets
    )

    bcs_MM = [dolfinx.fem.dirichletbc(displacement_function, np.concatenate([dofs_interface, dofs_meniscus]))]

    #---------------------------------------------------------------------------------------------------#
    # set other boundary conditions

    sourrounding_facets = facet_tags.indices[
        facet_tags.values == Boundary.surrounding.value
    ]

    symmetry_axis_facets = facet_tags.indices[
            facet_tags.values == Boundary.symmetry_axis.value
        ]

    crucible_surface_facets = facet_tags.indices[
            facet_tags.values == Surface.crucible.value
        ]

    #---------------------------------------------------------------------------------------------------#
    dirichlet_facets = np.concatenate([sourrounding_facets, crucible_surface_facets])

    dofs_hom_dirichlet = dolfinx.fem.locate_dofs_topological(
        displacement_function.function_space, 1, dirichlet_facets
    )

    value_MM = dolfinx.fem.Function(displacement_function.function_space)
    with value_MM.vector.localForm() as loc:
        loc.set(0)
    bcs_MM.append(dolfinx.fem.dirichletbc(value_MM, dofs_hom_dirichlet))

    #---------------------------------------------------------------------------------------------------#

    dirichlet_x_facets = symmetry_axis_facets
    dofs_symmetry_axis = dolfinx.fem.locate_dofs_topological(
        displacement_function.function_space.sub(0),
        1,
        dirichlet_x_facets,
    )
    
    bcs_MM.append(dolfinx.fem.dirichletbc(
            PETSc.ScalarType(0), dofs_symmetry_axis, displacement_function.function_space.sub(0)
        )
    )

    #---------------------------------------------------------------------------------------------------#
    # setup and solve mesh move problem
    laplace_problem.assemble(form_MM, bcs_MM)
    laplace_problem.solve()    

    return laplace_problem.solution

def mesh_move(mesh, displacement):
    mesh.geometry.x[:, :mesh.geometry.dim] += displacement.x.array.reshape((-1, mesh.geometry.dim)).real
    

def interface_displacement(displacement_function, normal_velocity_vector, v_pull_vector, Dt, beta, function_space, interface, meniscus, facet_tags, ax, fig):
    disp_vector = displacement_vector(normal_velocity_vector, v_pull_vector, Dt, beta, function_space, interface, meniscus, facet_tags, ax, fig)

    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_melt_crystal_interface = dolfinx.fem.locate_dofs_topological(
    displacement_function.function_space, 1, interface_facets 
    )

    with displacement_function.vector.localForm() as loc:
        values = loc.getArray()
        values[2 * dofs_melt_crystal_interface] = disp_vector[:,0]
        values[2 * dofs_melt_crystal_interface + 1] = disp_vector[:,1]
        loc.setArray(values)


def normal_velocity(velocity_vector, normals):
    # Projection of the velocity vector into the normal direction
    normal_projection = np.diag(normals @ velocity_vector.T).reshape(-1,1)
    normal_velocity_vector = np.repeat(normal_projection, 3, axis = 1) * normals
    
    return normal_velocity_vector

def displacement_vector(normal_velocity_vector, v_pull_vector, Dt, beta, function_space, interface, meniscus, facet_tags, ax, fig):
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        function_space, 1, interface_facets
    )

    #---------------------------------------------------------------------------------------------------#

    meniscus_facets = facet_tags.indices[
        facet_tags.values == meniscus.value
    ]

    dofs_meniscus = dolfinx.fem.locate_dofs_topological(
        function_space, 1, meniscus_facets
    )

    #---------------------------------------------------------------------------------------------------#
    
    coordinates_interface = function_space.tabulate_dof_coordinates()[dofs_interface]

    # permutation in ascending order (x-Coordinate)
    permutation_interface = np.argsort(coordinates_interface[:, 0])
    
    inverse_permutation = np.empty(permutation_interface.size, dtype=np.int32)
    for i in np.arange(permutation_interface.size):
        inverse_permutation[permutation_interface[i]] = i

    # permute coordinates and velocities in the same way
    coordinates_interface = coordinates_interface[permutation_interface]
    normal_velocity_vector = normal_velocity_vector[permutation_interface]
    v_pull_vector = v_pull_vector[permutation_interface]

    # calculate intermediate moved interface coordinates
    # x^{n + 1/2} = x^{n} + Dt * (v_n - v_p)
    coordinates_intermidiate_moved_interface = coordinates_interface + Dt * (normal_velocity_vector - v_pull_vector)

    # calculate tangent vector at end of intermediate moved interface
    tangent_intermidiate_moved_interface = coordinates_intermidiate_moved_interface[-1,:] - coordinates_intermidiate_moved_interface[-2,:]
    tangent_intermidiate_moved_interface /= np.repeat(np.linalg.norm(tangent_intermidiate_moved_interface), 3)

    #---------------------------------------------------------------------------------------------------#
    # Compute the growing angle from angle with meniscus (theta) and material specific angle (beta)

    # Theta: Angle between meniscus tangent and z-axis
    coordinates_meniscus = function_space.tabulate_dof_coordinates()[dofs_meniscus]

    # permutation in decresing order (y-Coordinate)
    permutation_meniscus = np.flipud(np.argsort(coordinates_meniscus[:, 1]))

    coordinates_meniscus = coordinates_meniscus[permutation_meniscus]

    # calculate tangent direction from first two points
    tangent_meniscus = coordinates_meniscus[1,:] - coordinates_meniscus[0,:]
    tangent_meniscus /= np.repeat(np.linalg.norm(tangent_meniscus), 3)

    theta = np.arccos(np.dot(tangent_meniscus, np.array([0.0, -1.0, 0.0]))) 
    # check that theta is in measured in mathematically correct rotaion
    theta *= np.sign(tangent_meniscus[0]) # check if sign of x component

    # Growing angle psi = theta - beta
    psi = theta - beta

    # compute growing direction [sin(psi), -cos(psi), 0.0]
    growing_normal = np.array([np.sin(psi), -np.cos(psi), 0.0])

    #---------------------------------------------------------------------------------------------------#
    # Compute intersection of the intermediate interface tangent with the growing direction
    A = np.array([tangent_intermidiate_moved_interface[0:-1], -growing_normal[0:-1]]).T.real
    b = coordinates_interface[-1,0:-1].real - coordinates_intermidiate_moved_interface[-1,0:-1].real
    x = solve(A,b)
    

    #move the intermidiate triple point to intersection
    coordinates_intermidiate_moved_interface[-1,:] += x[0] * tangent_intermidiate_moved_interface
    #---------------------------------------------------------------------------------------------------#
    # Compute moved interface by x^{n+1} = x^{n + 1/2} + Dt * v_p

    coordinates_moved_interface = coordinates_intermidiate_moved_interface + Dt * v_pull_vector

    coordinates_moved_interface = distribute_coordinates(coordinates_moved_interface, coordinates_interface)

    displacement_vector = coordinates_moved_interface - coordinates_interface

    ax.plot(coordinates_moved_interface[:, 0].real, coordinates_moved_interface[:, 1].real, '-k.')
    # ax.plot(coordinates_interface[:,0].real, coordinates_interface[:,1].real, '-r')

    return displacement_vector[inverse_permutation]

def meniscus_shape(z):
    h = z[0]
    
    gamma = 0.56
    rho = 6980.0
    g = 9.81

    l_c = (gamma / (rho * g) )**0.5

    x = l_c * (
        np.arccosh(2 * l_c/z) - np.arccosh(2*l_c/h)
    ) - l_c * (
        (4 - z**2/l_c**2)**0.5 - (4 - h**2/l_c**2)**0.5
    )

    return x

def meniscus_displacement(displacement_function, meniscus, facet_tags, ax, fig):
    
    meniscus_facets = facet_tags.indices[
        facet_tags.values == meniscus.value
    ]

    dofs_meniscus = dolfinx.fem.locate_dofs_topological(
        displacement_function.function_space, 1, meniscus_facets
    )

    #---------------------------------------------------------------------------------------------------#
    
    coordinates_meniscus = displacement_function.function_space.tabulate_dof_coordinates()[dofs_meniscus]

    # permutation in decresing order (y-Coordinate)
    permutation_meniscus = np.flipud(np.argsort(coordinates_meniscus[:, 1]))
    inverse_permutation = np.empty(permutation_meniscus.size, dtype=np.int32)
    for i in np.arange(permutation_meniscus.size):
        inverse_permutation[permutation_meniscus[i]] = i

    coordinates_meniscus = coordinates_meniscus[permutation_meniscus]

    dof_triple_point = dofs_meniscus[permutation_meniscus][0]
    dof_crucible = dofs_meniscus[permutation_meniscus][-1]

    displacement_of_triple_point = displacement_function.vector.getArray()[2*dof_triple_point:2*dof_triple_point+2].real
    
    new_height = coordinates_meniscus[0,1] + displacement_of_triple_point[1] - coordinates_meniscus[-1,1]

    meniscus_y_coordinates = np.linspace(0.0, new_height, coordinates_meniscus.shape[0])[::-1]
    meniscus_y_coordinates[-1] += 1e-10 # avoid division by 0
    meniscus_x_coordinates = meniscus_shape(meniscus_y_coordinates)
    
    meniscus_x_coordinates += coordinates_meniscus[0,0]
    meniscus_y_coordinates += coordinates_meniscus[-1,1]

    coordinates_new_meniscus = np.zeros_like(coordinates_meniscus)
    coordinates_new_meniscus[:,0] = meniscus_x_coordinates
    coordinates_new_meniscus[:,1] = meniscus_y_coordinates

    #---------------------------------------------------------------------------------------------------#
    # Fix last point to old point
    coordinates_new_meniscus[-1,:] = coordinates_meniscus[-1,:]
    #---------------------------------------------------------------------------------------------------#
    # TODO: Distribute coordinates along the graph (linear/exponential)
    coordinates_new_meniscus = distribute_coordinates(coordinates_new_meniscus, coordinates_meniscus)

    ax.plot(coordinates_new_meniscus[:, 0].real, coordinates_new_meniscus[:, 1].real, '-k.')
    ax.plot(coordinates_new_meniscus[0, 0].real, coordinates_new_meniscus[0, 1].real, '-r.')
    # ax.plot(coordinates_meniscus[:,0].real, coordinates_meniscus[:,1].real, '-r')
    #---------------------------------------------------------------------------------------------------#
    displacement_meniscus = - coordinates_meniscus
    displacement_meniscus += coordinates_new_meniscus

    displacement_meniscus = displacement_meniscus[inverse_permutation]

    # remove the displacement on triple point and meniscus
    cleaned_dofs_meniscus = dofs_meniscus[np.logical_and(dofs_meniscus != dof_triple_point, dofs_meniscus != dof_crucible)]
    displacement_meniscus = displacement_meniscus[np.logical_and(dofs_meniscus != dof_triple_point, dofs_meniscus != dof_crucible)]
    
    # write meniscus displacement in displacement function
    with displacement_function.vector.localForm() as loc:
        values = loc.getArray()
        values[2 * cleaned_dofs_meniscus] = displacement_meniscus[:,0]
        values[2 * cleaned_dofs_meniscus + 1] = displacement_meniscus[:,1]
        loc.setArray(values)


def distribute_coordinates(new_coordinates, old_coordinates):
    # coordinates must be ordered
    moved_coordinates = []
    
    new_taus = calculate_graph_coordinates(new_coordinates)
    old_taus = calculate_graph_coordinates(old_coordinates)
    # print("new")
    # print(new_taus)
    # print("old")
    # print(old_taus)
    dim = old_coordinates.shape[1]
        
    # permutate coordinates so that taus are in ascending order
    permutated_coordinates = new_coordinates#new_coordinates[permutation]
    new_taus = new_taus#new_taus[permutation]

    # create piecewise linear function for the coordinate
    cond_list = [np.isclose(old_taus, 0.0)]
    for i in range(len(new_taus) - 1):
        cond_list.append(
            np.logical_and(
                new_taus[i] < old_taus,
                old_taus <= new_taus[i+1]
            )
        )
    
    cond_list.append(np.isclose(old_taus, 1.0))
    # for i in range(len(cond_list)):
    #     print(f"Position {i+1}/{len(cond_list)}: {cond_list[i].any()}")
    #     print(cond_list[i])

    for j in range(dim):
        func_list = [permutated_coordinates[0, j] * np.ones(len(old_taus))]

        for i in range(len(new_taus) - 1):
            func_list.append(
                permutated_coordinates[i, j]
                + (permutated_coordinates[i + 1, j] - permutated_coordinates[i, j]) * (old_taus - new_taus[i]) / (new_taus[i + 1] - new_taus[i])
            )

        func_list.append(permutated_coordinates[-1, j] * np.ones(len(old_taus)))

        moved_coordinates.append(np.select(cond_list, func_list, default=np.inf))

    moved_coordinates = np.array(moved_coordinates)

    moved_coordinates = moved_coordinates.T

    # TODO: Not good, but I don#t find the problem why condlist fails.
    if len([moved_coordinates[:,0] == np.inf]) > 0:
        moved_coordinates[moved_coordinates[:,0] == np.inf] = old_coordinates[moved_coordinates[:,0] == np.inf]

    return moved_coordinates


def calculate_graph_coordinates(coordinates):
    """
    Parameterize the graph with specific graph coordinates $\tau \in \left[0,1 \right]$
    """   
    
    coordinates_of_dofs = coordinates.T

    number_of_coordinates = coordinates_of_dofs.shape[1]
    # calculate the normalized graph coordinate
    tau = [0.0]
    for i in range(number_of_coordinates - 1):
        tau.append(tau[-1] + np.linalg.norm(coordinates_of_dofs[:, i] - coordinates_of_dofs[:, i+1]))

    tau = np.array(tau)
    if len(tau) > 1:
        tau = tau / tau[-1]

    return tau