import dolfinx
import ufl

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt

from ..equations.laplace import Laplace
#####################################################################################################
#                                                                                                   #
#                                     TEMPERATURE SCALING                                           #
#                                                                                                   #
#####################################################################################################

def set_temperature_scaling(heat_problem, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat, bcs_T, desired_temp, interface, facet_tags):
    
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        heat_problem.solution.function_space, 1, interface_facets
    )

    coordinates_interface = heat_problem.solution.function_space.tabulate_dof_coordinates()[dofs_interface]

    # permutation in ascending order (x-Coordinate)
    permutation_interface = np.argsort(coordinates_interface[:, 0])

    dof_triple_point = dofs_interface[permutation_interface][-1]

    #---------------------------------------------------------------------------------------------------#

    def calculate_temp_diff(temp_scaling):
        if type(temp_scaling) == np.ndarray:
            temp_scaling = temp_scaling.real[0]
        
        heat_problem._heat_scaling = temp_scaling

        heat_form = heat_problem.setup(heat_problem.solution, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat)
        heat_problem.assemble(heat_form, bcs_T) 
        heat_problem.solve()

        with heat_problem.solution.vector.localForm() as loc:
            heat_value_triple_point = loc.getArray()[dof_triple_point].real
        
        return heat_value_triple_point - desired_temp

    heat_scaling = optimize.newton(calculate_temp_diff, heat_problem._heat_scaling, tol=1e-7)
    
    if type(heat_scaling) == np.ndarray:
            heat_scaling = heat_scaling.real[0]
    
    heat_problem._heat_scaling = heat_scaling


#####################################################################################################
#                                                                                                   #
#                                      MESH MOVEMENT                                                #
#                                                                                                   #
#####################################################################################################

def mesh_move(function, Volume, Boundary, Surface, Interface, cell_tags, facet_tags):
    melt = Volume.melt
    crystal = Volume.crystal
    interface = Interface.melt_crystal
    
    # Setup mesh movement problem
    mesh = function.function_space.mesh
    
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
    
    Space_MM = dolfinx.FunctionSpace(mesh, vector_element)

    laplace_problem = Laplace(Space_MM)
    form_MM = laplace_problem.setup(laplace_problem.solution, dV, dA, dI)


    #---------------------------------------------------------------------------------------------------#
    # set function for displacement on melt-crystal interface
    displacement_function = dolfinx.Function(Space_MM)

    #---------------------------------------------------------------------------------------------------#
    # calculate displacement on melt crystal interface 

    # move wrong dofs in melt
    threshold_function = lambda value: value <= 505.

    marked_dofs = dofs_with_threshold(function, melt, cell_tags, threshold_function)
    old_interface_coordinates, new_interface_coordinates, moved_dofs = get_new_interface_coordinates(function, marked_dofs, interface, melt, facet_tags, cell_tags, threshold_function)
    moved_interface = project_graphs(old_interface_coordinates, new_interface_coordinates)
    
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        function.function_space, 1, interface_facets
    )

    if moved_interface != []:
        displacement = moved_interface - old_interface_coordinates

        with displacement_function.vector.localForm() as loc:
            values = loc.getArray()
            values[2 * moved_dofs] = displacement[:,0]
            values[2 * moved_dofs + 1] = displacement[:,1]
            loc.setArray(values)

    #---------------------------------------------------------------------------------------------------#
    # move wrong dofs in crystal
    threshold_function = lambda value: value > 505.

    marked_dofs = dofs_with_threshold(function, crystal, cell_tags, threshold_function)
    old_interface_coordinates, new_interface_coordinates, moved_dofs = get_new_interface_coordinates(function, marked_dofs, interface, crystal, facet_tags, cell_tags, threshold_function)
    moved_interface = project_graphs(old_interface_coordinates, new_interface_coordinates)
    
    if moved_interface != []:
        displacement = moved_interface - old_interface_coordinates


        with displacement_function.vector.localForm() as loc:
            values = loc.getArray()
            values[2 * moved_dofs] = displacement[:,0]
            values[2 * moved_dofs + 1] = displacement[:,1]
            loc.setArray(values)

    # set displacement as dirichlet BC
    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        Space_MM, 1, interface_facets
    )

    bcs_MM = [dolfinx.DirichletBC(displacement_function, dofs_interface)]

    #---------------------------------------------------------------------------------------------------#
    # set other boundary conditions

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

    melt_surface_facets = facet_tags.indices[
            facet_tags.values == Surface.melt.value
        ]

    #---------------------------------------------------------------------------------------------------#
    dirichlet_facets = np.concatenate([sourrounding_facets, crucible_surface_facets, crystal_surface_facets, melt_surface_facets])

    dofs_hom_dirichlet = dolfinx.fem.locate_dofs_topological(
        Space_MM, 1, dirichlet_facets
    )

    value_MM = dolfinx.Function(Space_MM)
    with value_MM.vector.localForm() as loc:
        loc.set(0)
    bcs_MM.append(dolfinx.DirichletBC(value_MM, dofs_hom_dirichlet))

    #---------------------------------------------------------------------------------------------------#

    dirichlet_x_facets = symmetry_axis_facets
    dofs_symmetry_axis = dolfinx.fem.locate_dofs_topological(
        (Space_MM.sub(0), Space_MM.sub(0).collapse(),),
        1,
        dirichlet_x_facets,
    )

    value_MM = dolfinx.Function(Space_MM.sub(0).collapse()) # only BC on x-component
    with value_MM.vector.localForm() as loc:
        loc.set(0)

    bcs_MM.append(dolfinx.DirichletBC(
            value_MM, dofs_symmetry_axis, Space_MM.sub(0)
        )
    )


    #---------------------------------------------------------------------------------------------------#
    # setup and solve mesh move problem
    laplace_problem.assemble(form_MM, bcs_MM)
    laplace_problem.solve()    

    return laplace_problem.solution


def dofs_with_threshold(function, volume, cell_tags, threshold_function):
    # Get the dofs from a volume/subset which satisfy some threshold function
    volume_cells = cell_tags.indices[
        cell_tags.values == volume.value
    ]

    dofs_volume = dolfinx.fem.locate_dofs_topological(
        function.function_space, 2, volume_cells
    )

    with function.vector.localForm() as loc:
        local_Values = loc.getValues(dofs_volume).real
    
    return dofs_volume[threshold_function(local_Values)]

def get_new_interface_coordinates(function, marked_dofs, interface, volume, facet_tags, cell_tags, threshold_function):
    
    volume_cells = cell_tags.indices[
        cell_tags.values == volume.value
    ]

    dofs_volume = dolfinx.fem.locate_dofs_topological(
        function.function_space, 2, volume_cells
    )

    interface_facets = facet_tags.indices[
        facet_tags.values == interface.value
    ]

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        function.function_space, 1, interface_facets
    )

    #---------------------------------------------------------------------------------------------------#
    # get dofs that need to be moved
    dofs_to_move_on_interface = dofs_interface[[dof in marked_dofs for dof in dofs_interface]]
    #---------------------------------------------------------------------------------------------------#
    
    bb_tree = dolfinx.cpp.geometry.BoundingBoxTree(function.function_space.mesh, function.function_space.mesh.geometry.dim)
    cell_ids = np.array([])
    

    for point in function.function_space.tabulate_dof_coordinates()[marked_dofs]: #function.function_space.mesh.geometry.x[marked_dofs]
        cell_candidates = dolfinx.cpp.geometry.compute_collisions(bb_tree, point)
        cell_ids = np.concatenate((cell_ids, cell_candidates), axis = None)
    
    cell_ids = np.unique(cell_ids).astype(int)
    
    interface_coords = []

    function_values = function.compute_point_values().real

    coordinates = function.function_space.mesh.geometry.x
    threshold = 505.

    for cell_id in cell_ids:
        cell = function.function_space.dofmap.cell_dofs(cell_id)
        dofs = cell.astype(np.int32)

        # check if all dofs are from the subset
        if np.all(np.in1d(dofs, dofs_volume)):
            values = function_values[dofs].reshape(-1,)
            # Check if the interface is cutting the cell
            if bool(any([val >= threshold for val in values]) * any([val < threshold for val in values])):
                threshold_function_bool = threshold_function(values)
                coordinates_of_dofs = coordinates[dofs]

                violation_values = values[threshold_function_bool]
                violation_coordinates = coordinates_of_dofs[threshold_function_bool]

                no_violation_values = values[~threshold_function_bool]
                no_violation_coordinates = coordinates_of_dofs[~threshold_function_bool]
                #change for higher polynomial degree
                for v_value, v_coord in zip(violation_values, violation_coordinates):
                    for n_v_value, n_v_coord  in zip(no_violation_values, no_violation_coordinates):
                        new_coord = n_v_coord + (threshold - n_v_value) / (v_value - n_v_value) * (
                            v_coord - n_v_coord
                        )
                        interface_coords.append(new_coord)
    
    interface_coords = np.unique(np.array(interface_coords), axis=0)
    
    if not interface_coords.size == 0: 
        interface_coords = interface_coords[np.argsort(interface_coords[:, 0])]
    
    if len(dofs_to_move_on_interface) > 0:
        old_interface_coordinates = coordinates[dofs_to_move_on_interface]
        min_x_coord_on_old_interface = old_interface_coordinates[:,0].min()
        max_x_coord_on_old_interface = old_interface_coordinates[:,0].max()

    # TODO: the last/first coordinate is wrong somehow:
    # if interface_coords.shape[0] > 1 and over_threshold:
    #     if not np.isclose(min_x_coord_on_old_interface, interface_coords[0, 0]):
    #         interface_coords = interface_coords[:-1,:]
    # if interface_coords.shape[0] > 1 and not over_threshold:
    #     interface_coords = interface_coords[1:,:]
    if interface_coords.shape[0] > 1 and len(dofs_to_move_on_interface) > 0:
        if not np.isclose(min_x_coord_on_old_interface, interface_coords[0, 0]):
            interface_coords = interface_coords[1:,:]

        if not np.isclose(max_x_coord_on_old_interface, interface_coords[-1, 0]):
            interface_coords = interface_coords[:-1,:]
    
    if interface_coords != []:
        fig, ax = plt.subplots(1,1)
        ax.plot(interface_coords[:,0], interface_coords[:,1])
        ax.plot(coordinates[dofs_interface][:,0], coordinates[dofs_interface][:,1])
        fig.savefig("interface.png")

    return coordinates[dofs_interface], interface_coords, dofs_to_move_on_interface

def calculate_graph_coordinates(coordinates):
    """
    Parameterize the graph with specific graph coordinates $\tau \in \left[0,1 \right]$
    """
    permutation = np.argsort(coordinates[:, 0])    
    
    inverse_permutation = np.empty(permutation.size, dtype=np.int32)
    for i in np.arange(permutation.size):
        inverse_permutation[permutation[i]] = i
    
    
    coordinates_of_dofs = coordinates[permutation].T
    
    number_of_coordinates = coordinates_of_dofs.shape[1]
    # calculate the normalized graph coordinate
    tau = [0.0]
    for i in range(number_of_coordinates - 1):
        tau.append(tau[-1] + np.linalg.norm(coordinates_of_dofs[:, i] - coordinates_of_dofs[:, i+1]))

    tau = np.array(tau) / tau[-1]

    return tau[inverse_permutation]

def project_graphs(old_coordinates, new_coordinates):
    """
    Project the old graph on the new graph by their specific graph coordinate.
    """
    moved_coordinates = []

    if new_coordinates != []:
        dim = old_coordinates.shape[1]
        
        old_taus = calculate_graph_coordinates(old_coordinates)
        new_taus = calculate_graph_coordinates(new_coordinates)

        permutation = np.argsort(new_coordinates[:, 0])
            
        # permutate coordinates so that taus are in ascending order
        permutated_coordinates = new_coordinates[permutation]
        new_taus = new_taus[permutation]
        
        # create piecewise linear function for the coordinate
        cond_list = [np.isclose(old_taus, 0.0)]
        for i in range(len(new_taus) - 2):
            cond_list.append(
                np.logical_and(
                    new_taus[i] < old_taus, 
                    old_taus <= new_taus[i + 1]
                )
            )
        cond_list.append(np.isclose(old_taus, 1.0))
        
        for j in range(dim):
            func_list = [permutated_coordinates[0, j] * np.ones(len(old_taus))]

            for i in range(len(new_taus) - 2):
                func_list.append(
                    permutated_coordinates[i, j]
                    + (permutated_coordinates[i + 1, j] - permutated_coordinates[i, j]) * (old_taus - new_taus[i]) / (new_taus[i + 1] - new_taus[i])
                )

            func_list.append(permutated_coordinates[-1, j] * np.ones(len(old_taus)))

            moved_coordinates.append(np.select(cond_list, func_list))
            
        moved_coordinates = np.array(moved_coordinates)

        moved_coordinates = moved_coordinates.T

        fig, ax = plt.subplots(1,1)
        ax.plot(old_coordinates[:,0],old_coordinates[:,1], '-ro')
        ax.plot(new_coordinates[:, 0], new_coordinates[:, 1], '--go')
        ax.plot(moved_coordinates[:, 0], moved_coordinates[:, 1], '--bo')
        # ax.set_aspect('equal', 'box')
        fig.savefig("interface.png")

    return moved_coordinates