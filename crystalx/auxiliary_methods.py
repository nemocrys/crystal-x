from math import modf

import numpy as np
from scipy.linalg import solve
import dolfinx
from geometry.geometry import Interface
import ufl
from mpi4py import MPI

def timeToStr(totalTime):
    millis, sec = modf(totalTime)
    millis = round(millis, 3) * 1000
    sec, millis = int(sec), int(millis)
    if sec < 1:
        return f"{millis}ms"
    elif sec < 60.0:
        return f"{sec}s {millis}ms"
    else:
        return f"{sec // 60}min {(sec % 60)}s {millis}ms"

def project(function, functionspace, **kwargs):
    """
    Computes the L2-projection of the function into the functionspace.
    See: https://fenicsproject.org/qa/6832/what-is-the-difference-between-interpolate-and-project/
    """
    if "name" in kwargs.keys():
        assert isinstance(kwargs["name"], str)
        sol = dolfinx.Function(functionspace, name = kwargs["name"])
    else:
        sol = dolfinx.Function(functionspace)
    
    w = ufl.TrialFunction(functionspace)
    v = ufl.TestFunction(functionspace)

    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(function, v) * ufl.dx

    problem = dolfinx.fem.LinearProblem(a, L, bcs=[])
    sol.interpolate(problem.solve())

    return sol


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

def normal_velocity(velocity_vector, normals):
    # Projection of the velocity vector into the normal direction
    normal_projection = np.diag(normals @ velocity_vector.T).reshape(-1,1)
    normal_velocity_vector = np.repeat(normal_projection, 3, axis = 1) * normals
    
    return normal_velocity_vector

def interface_displacement(normal_velocity_vector, v_pull_vector, Dt, beta, function_space, interface, meniscus, facet_tags):
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
    
    displacement_vector = coordinates_moved_interface - coordinates_interface
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

def meniscus_displacement(displacement_function, function_space, meniscus, facet_tags):
    
    meniscus_facets = facet_tags.indices[
        facet_tags.values == meniscus.value
    ]

    dofs_meniscus = dolfinx.fem.locate_dofs_topological(
        function_space, 1, meniscus_facets
    )

    #---------------------------------------------------------------------------------------------------#
    
    coordinates_meniscus = function_space.tabulate_dof_coordinates()[dofs_meniscus]

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

    #---------------------------------------------------------------------------------------------------#

    displacement_meniscus = - coordinates_meniscus
    displacement_meniscus[:, 0] += meniscus_x_coordinates
    displacement_meniscus[:, 1] += meniscus_y_coordinates

    # mark triple point and crucible point
    displacement_meniscus[0, 2] = np.inf
    displacement_meniscus [-1,2] = np.inf

    displacement_meniscus = displacement_meniscus[inverse_permutation]

    # remove the displacement on triple point and meniscus
    cleaned_dofs_meniscus = dofs_meniscus[np.logical_and(dofs_meniscus != dof_triple_point, dofs_meniscus != dof_crucible)]
    displacement_meniscus = displacement_meniscus[displacement_meniscus[:,2] != np.inf]
    
    # write meniscus displacement in displacement function
    with displacement_function.vector.localForm() as loc:
        values = loc.getArray()
        values[2 * cleaned_dofs_meniscus] = displacement_meniscus[:,0]
        values[2 * cleaned_dofs_meniscus + 1] = displacement_meniscus[:,1]
        loc.setArray(values)
