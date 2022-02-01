import dolfinx

import numpy as np
from scipy import optimize

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

def mesh_move(function, volume, interface, cell_tags, facet_tags):
    threshold_function = lambda value: value <= 505.

    marked_dofs = dofs_with_threshold(function, volume, cell_tags, threshold_function)
    get_new_interface_coordinates(function, marked_dofs, interface, volume, facet_tags, cell_tags, threshold_function)

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

    dofs_to_move_on_interface = np.array(list(set(dofs_interface).intersection(set(marked_dofs))))

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
    
    print(interface_coords.shape)
    exit()
    return interface_coords



