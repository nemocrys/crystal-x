import dolfinx
import ufl

import numpy as np
from scipy import optimize, interpolate
from petsc4py import PETSc

import matplotlib.pyplot as plt

from .equations.laplace import Laplace
#####################################################################################################
#                                                                                                   #
#                                     TEMPERATURE SCALING                                           #
#                                                                                                   #
#####################################################################################################

def set_temperature_scaling(heat_problem, dV, dA, dI, rho, kappa, omega, varsigma, h,  T_amb, A, f_heat, bcs_T, desired_temp, interface, facet_tags):
    
    interface_facets = facet_tags.find(
        interface.value
    )

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

def interface_displacement(function, T_melt, Volume, Boundary, Surface, Interface, cell_tags, facet_tags):
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
    
    Space_MM = dolfinx.fem.FunctionSpace(mesh, vector_element)

    laplace_problem = Laplace(Space_MM)
    form_MM = laplace_problem.setup(laplace_problem.solution, dV, dA, dI)


    #---------------------------------------------------------------------------------------------------#
    # set function for displacement on melt-crystal interface
    displacement_function = dolfinx.fem.Function(Space_MM)
    
    threshold_function_tol = 1e-4
    interface_computation_tol = threshold_function_tol
    #---------------------------------------------------------------------------------------------------#
    
    threshold_function_melt = lambda value: value < (T_melt - threshold_function_tol)
    threshold_function_crystal = lambda value: value > (T_melt + threshold_function_tol)

    #---------------------------------------------------------------------------------------------------#
    # calculate displacement on melt crystal interface 

    # move wrong dofs in melt

    marked_dofs = dofs_with_threshold(function, melt, cell_tags, threshold_function_melt)
    old_interface_coordinates_melt, new_interface_coordinates_melt, moved_dofs_melt = get_new_interface_coordinates(function, T_melt, marked_dofs, interface, melt, facet_tags, cell_tags, [threshold_function_melt, threshold_function_crystal], interface_computation_tol)
    # moved_interface = project_graphs(old_interface_coordinates, new_interface_coordinates, melt)
    # moved_interface = project_graphs_new(old_interface_coordinates, new_interface_coordinates, melt)
    # moved_interface_melt = moved_interface

    # interface_facets = facet_tags.find(
    #     interface.value
    # )

    # dofs_interface = dolfinx.fem.locate_dofs_topological(
    #     function.function_space, 1, interface_facets
    # )

    # if moved_interface != []:
    #     displacement = moved_interface - old_interface_coordinates
    #     with displacement_function.vector.localForm() as loc:
    #         values = loc.getArray()
    #         values[2 * moved_dofs] = displacement[:,0]
    #         values[2 * moved_dofs + 1] = displacement[:,1]
    #         loc.setArray(values)

    #---------------------------------------------------------------------------------------------------#

    # move wrong dofs in crystal

    marked_dofs = dofs_with_threshold(function, crystal, cell_tags, threshold_function_crystal)
    old_interface_coordinates_crystal, new_interface_coordinates_crystal, moved_dofs_crystal = get_new_interface_coordinates(function, T_melt, marked_dofs, interface, crystal, facet_tags, cell_tags, [threshold_function_crystal, threshold_function_melt], interface_computation_tol)
    # # moved_interface = project_graphs(old_interface_coordinates, new_interface_coordinates, crystal)
    # moved_interface = project_graphs_new(old_interface_coordinates, new_interface_coordinates, crystal)
    # moved_interface_crystal = moved_interface

    # if moved_interface != []:
    #     displacement = moved_interface - old_interface_coordinates
    #     with displacement_function.vector.localForm() as loc:
    #         values = loc.getArray()
    #         values[2 * moved_dofs] = displacement[:,0]
    #         values[2 * moved_dofs + 1] = displacement[:,1]
    #         loc.setArray(values)
    #---------------------------------------------------------------------------------------------------#
    
    print("moved_dofs")
    print(moved_dofs_melt.shape, moved_dofs_crystal.shape)
    moved_dofs = np.concatenate((moved_dofs_melt, moved_dofs_crystal))
    print(moved_dofs.shape)

    #---------------------------------------------------------------------------------------------------#

    interface_facets = facet_tags.find(
        interface.value
    )

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        Space_MM, 1, interface_facets
    )
    
    unmoved_dofs = np.setxor1d(dofs_interface, moved_dofs)
    unmoved_coordinates = Space_MM.mesh.geometry.x[unmoved_dofs].reshape(-1,3)
    #---------------------------------------------------------------------------------------------------#
    print("old_interface_coordinates")
    print(old_interface_coordinates_melt.shape, old_interface_coordinates_crystal.shape, unmoved_coordinates.shape)
    # old_interface_coordinates = np.concatenate((old_interface_coordinates_melt, old_interface_coordinates_crystal, unmoved_coordinates))
    old_interface_coordinates = np.concatenate((old_interface_coordinates_melt, old_interface_coordinates_crystal))

    print(old_interface_coordinates.shape, dofs_interface.shape)
    
    print("new_interface_coordinates")
    print(new_interface_coordinates_melt.shape, new_interface_coordinates_crystal.shape, unmoved_coordinates.shape)
    # new_interface_coordinates = np.concatenate((new_interface_coordinates_melt, new_interface_coordinates_crystal, unmoved_coordinates))
    new_interface_coordinates = np.concatenate((new_interface_coordinates_melt, new_interface_coordinates_crystal))
    print(new_interface_coordinates.shape, dofs_interface.shape)
    #---------------------------------------------------------------------------------------------------#

    fig, ax = plt.subplots(1,1)
    coord_moved = Space_MM.mesh.geometry.x[moved_dofs].reshape(-1,3)
    points, = ax.plot(coord_moved[:, 0], coord_moved[:, 1], 'or')
    old_points, = ax.plot(unmoved_coordinates[:, 0], unmoved_coordinates[:, 1], 'ob')
    ax.legend([old_points, points], ["unmoved dofs", "moved dofs"])
    # ax.set_xlim([0.0, 0.004])
    fig.savefig(f"dofs.png")
    
    #---------------------------------------------------------------------------------------------------#

    moved_interface = project_graphs_new(old_interface_coordinates, new_interface_coordinates)

    if moved_interface != []:
        displacement = moved_interface - old_interface_coordinates
        with displacement_function.vector.localForm() as loc:
            values = loc.getArray()
            # values[2 * dofs_interface] = displacement[:,0]
            # values[2 * dofs_interface + 1] = displacement[:,1]
            values[2 * moved_dofs] = displacement[:,0]
            values[2 * moved_dofs + 1] = displacement[:,1]
            loc.setArray(values)
    # #---------------------------------------------------------------------------------------------------#
    # if moved_interface_crystal != [] and moved_interface_melt != []:
    #     fig, ax = plt.subplots(1,1)
    #     p_moved_melt, = ax.plot(moved_interface_melt[:, 0], moved_interface_melt[:, 1], '--go')
    #     p_moved_crystal, = ax.plot(moved_interface_crystal[:, 0], moved_interface_crystal[:, 1], '--ro')
    #     # p_old, = ax.plot(old_coordinates[:,0],old_coordinates[:,1], '--r^')
    #     # ax.set_aspect('equal', 'box')
    #     ax.legend([p_moved_melt, p_moved_crystal], ["Melt", "Crystal"])
    #     ax.set_xlim([0.0, 0.004])
    #     fig.savefig(f"interface_combined.png")

    #---------------------------------------------------------------------------------------------------#
    # set displacement as dirichlet BC
    interface_facets = facet_tags.find(
        interface.value
    )

    dofs_interface = dolfinx.fem.locate_dofs_topological(
        Space_MM, 1, interface_facets
    )

    bcs_MM = [dolfinx.fem.dirichletbc(displacement_function, dofs_interface)]

    #---------------------------------------------------------------------------------------------------#
    # set other boundary conditions

    sourrounding_facets = facet_tags.find(
        Boundary.surrounding.value
    )

    symmetry_axis_facets = facet_tags.find(
            Boundary.symmetry_axis.value
    )

    crystal_surface_facets = facet_tags.find(
            Surface.crystal.value
    )

    crucible_surface_facets = facet_tags.find(
            Surface.crucible.value
    )

    meniscus_surface_facets = facet_tags.find(
            Surface.meniscus.value
    )

    melt_flat_surface_facets = facet_tags.find(
            Surface.melt_flat.value
    )
    #---------------------------------------------------------------------------------------------------#
    dirichlet_facets = np.concatenate([sourrounding_facets, crucible_surface_facets, crystal_surface_facets, meniscus_surface_facets, melt_flat_surface_facets])

    dofs_hom_dirichlet = dolfinx.fem.locate_dofs_topological(
        Space_MM, 1, dirichlet_facets
    )

    value_MM = dolfinx.fem.Function(Space_MM)
    with value_MM.vector.localForm() as loc:
        loc.set(0)
    bcs_MM.append(dolfinx.fem.dirichletbc(value_MM, dofs_hom_dirichlet))

    #---------------------------------------------------------------------------------------------------#

    dirichlet_x_facets = symmetry_axis_facets
    dofs_symmetry_axis = dolfinx.fem.locate_dofs_topological(
        Space_MM.sub(0),
        1,
        dirichlet_x_facets,
    )

    bcs_MM.append(dolfinx.fem.dirichletbc(
            PETSc.ScalarType(0), dofs_symmetry_axis, Space_MM.sub(0)
        )
    )

    #---------------------------------------------------------------------------------------------------#
    # setup and solve mesh move problem
    laplace_problem.assemble(form_MM, bcs_MM)
    laplace_problem.solve()    

    return laplace_problem.solution

def mesh_move(mesh, displacement):
    mesh.geometry.x[:, :mesh.geometry.dim] += displacement.x.array.reshape((-1, mesh.geometry.dim)).real


def dofs_with_threshold(function, volume, cell_tags, threshold_function):
    # Get the dofs from a volume/subset which satisfy some threshold function
    volume_cells = cell_tags.indices[
        cell_tags.values == volume.value
    ]

    dofs_volume = dolfinx.fem.locate_dofs_topological(
        function.function_space, 2, volume_cells
    )

    return dofs_volume[threshold_function(function.x.array[dofs_volume].real)]

def get_new_interface_coordinates(function, T_melt, marked_dofs, interface, volume, facet_tags, cell_tags, threshold_functions, tolarance):
    TOL = tolarance
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
    bb_tree = dolfinx.geometry.BoundingBoxTree(function.function_space.mesh, function.function_space.mesh.geometry.dim)
    cell_ids = np.array([])
    

    for point in function.function_space.tabulate_dof_coordinates()[marked_dofs]: #function.function_space.mesh.geometry.x[marked_dofs]
        cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, point)
        cell_ids = np.concatenate((cell_ids, cell_candidates.links(0)), axis = None)
    
    cell_ids = np.unique(cell_ids).astype(int)
    
    interface_coords = []

    value_space = dolfinx.fem.FunctionSpace(function.function_space.mesh, ("CG",1))
    values = dolfinx.fem.Function(value_space)
    values.interpolate(function)
    function_values = values.x.array.real
        
    coordinates = function.function_space.mesh.geometry.x
    threshold = T_melt

    for cell_id in cell_ids:
        cell = function.function_space.dofmap.cell_dofs(cell_id)
        dofs = cell.astype(np.int32)

        # check if all dofs are from the subset
        if np.all(np.in1d(dofs, dofs_volume)):
            values = function_values[dofs].reshape(-1,)
            
            # Check if the interface is cutting the cell
            if bool(any(threshold_functions[0](values)) * any(threshold_functions[1](values))):
                threshold_function_bool_violation = threshold_functions[0](values)
                threshold_function_bool_no_violation = threshold_functions[1](values)
                coordinates_of_dofs = coordinates[dofs]

                violation_values = values[threshold_function_bool_violation]
                violation_coordinates = coordinates_of_dofs[threshold_function_bool_violation]

                no_violation_values = values[threshold_function_bool_no_violation]
                no_violation_coordinates = coordinates_of_dofs[threshold_function_bool_no_violation]
                
                if bool(any(threshold_function_bool_violation * threshold_function_bool_no_violation)):
                    print(threshold_function_bool_violation, threshold_function_bool_no_violation)
                    print(violation_values, no_violation_values)
                    exit()

                if len(violation_values) + len(no_violation_values) == 3:

                    for v_value, v_coord in zip(violation_values, violation_coordinates):
                        for n_v_value, n_v_coord  in zip(no_violation_values, no_violation_coordinates):
                            new_coord = n_v_coord + (threshold - n_v_value) / (v_value - n_v_value) * (
                                v_coord - n_v_coord
                            )
                            if new_coord[0] < 0.0:
                                # print(n_v_coord, (threshold - n_v_value), (v_value - n_v_value), (threshold - n_v_value) / (v_value - n_v_value) , v_coord - n_v_coord)
                                print(threshold, violation_values-threshold, no_violation_values-threshold)
                                print(TOL)
                                print(abs(violation_values-threshold)>TOL, abs(no_violation_values-threshold)>TOL)
                                exit()
                            # print(violation_values, no_violation_values)
                            # print(abs(violation_values-threshold), abs(no_violation_values-threshold))
                            # print()
                            interface_coords.append(new_coord)

    interface_coords = np.array(interface_coords).reshape(-1,3)
    
    return coordinates[dofs_to_move_on_interface], interface_coords, dofs_to_move_on_interface

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

    tau = np.array(tau)
    if len(tau) > 1:
        tau = tau / tau[-1]

    return tau[inverse_permutation]

def project_graphs(old_coordinates, new_coordinates, volume):
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
        permutated_coordinates = new_coordinates#new_coordinates[permutation]
        new_taus = new_taus#new_taus[permutation]

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

            moved_coordinates.append(np.select(cond_list, func_list, default=np.inf))

        moved_coordinates = np.array(moved_coordinates)

        moved_coordinates = moved_coordinates.T

        # TODO: Not good, but I don#t find the problem why condlist fails.
        if len([moved_coordinates[:,0] == np.inf]) > 0:
            moved_coordinates[moved_coordinates[:,0] == np.inf] = old_coordinates[moved_coordinates[:,0] == np.inf]

        permutation = np.argsort(moved_coordinates[:, 0])
    
        fig, ax = plt.subplots(1,1)
        p_new, = ax.plot(new_coordinates[:, 0], new_coordinates[:, 1], '--go')
        p_moved, = ax.plot(moved_coordinates[:, 0], moved_coordinates[:, 1], '--r^')
        # p_old, = ax.plot(old_coordinates[:,0],old_coordinates[:,1], '--r^')
        # ax.set_aspect('equal', 'box')
        ax.legend([p_moved, p_new], ["Moved", "New"])
        ax.set_xlim([0.0, 0.004])
        fig.savefig(f"interface_displacement_{volume.name}.png")
        
        # print(volume.name)
        # print(new_coordinates.shape[0]-old_coordinates.shape[1])
        # print(new_coordinates.shape[0]-moved_coordinates.shape[0])

    return moved_coordinates

def project_graphs_new(old_coordinates, new_coordinates):
    """
    Project the old graph on the new graph by their specific graph coordinate.
    """
    moved_coordinates = []
    if len(new_coordinates) >= 2:
        old_permutation = np.argsort(old_coordinates[:, 0])
        new_permutation = np.argsort(new_coordinates[:, 0])

        inverse_permutation = np.empty(old_permutation.size, dtype=np.int32)
        for i in np.arange(old_permutation.size):
            inverse_permutation[old_permutation[i]] = i
        
        # permutate coordinates so that taus are in ascending order
        old_coordinates = old_coordinates[old_permutation]
        new_coordinates = new_coordinates[new_permutation]

        old_taus = calculate_graph_coordinates(old_coordinates)
        new_taus = calculate_graph_coordinates(new_coordinates)

        tau_to_x = interpolate.interp1d(new_taus, new_coordinates[:,0])
        tau_to_y = interpolate.interp1d(new_taus, new_coordinates[:,1])

        new_x = tau_to_x(old_taus)
        new_y = tau_to_y(old_taus)

        moved_coordinates = np.zeros((old_taus.shape[0], 3))
        moved_coordinates[:, 0] = new_x
        moved_coordinates[:, 1] = new_y

        fig, ax = plt.subplots(1,1)
        taus = new_taus#np.arange(0, 1, 0.001)
        x_int = tau_to_x(taus)
        y_int = tau_to_y(taus)
        points, = ax.plot(new_coordinates[:, 0], new_coordinates[:, 1], '^')
        # old_points, = ax.plot(old_coordinates[:, 0], old_coordinates[:, 1], 'x')
        interface, = ax.plot(x_int, y_int, '-r')
        # new_points, = ax.plot(moved_coordinates[:, 0], moved_coordinates[:, 1], 'o')
        # ax.legend([points, old_points, interface], ["Interpolation Points","Old interface","Interface"])
        # ax.legend([interface, new_points, old_points, points], ["interface","new dofs","old dofs","interpolation points"])
        # ax.set_xlim([0.0, 0.004])
        fig.savefig(f"interface.png")

        return moved_coordinates[inverse_permutation]
    
    return []

#####################################################################################################
#                                                                                                   #
#                                   FUNCTION EVALUATION                                             #
#                                                                                                   #
#####################################################################################################

def evaluate_function(function, points):
    bb_tree = dolfinx.geometry.BoundingBoxTree(function.function_space.mesh, function.function_space.mesh.topology.dim)

    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(function.function_space.mesh, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
     
    return function.eval(points_on_proc, cells)