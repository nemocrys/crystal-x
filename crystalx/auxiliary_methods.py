from math import modf

import numpy as np
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

    permutation = np.argsort(coordinates_interface[:, 0])
    
    inverse_permutation = np.empty(permutation.size, dtype=np.int32)
    for i in np.arange(permutation.size):
        inverse_permutation[permutation[i]] = i

    coordinates_interface = coordinates_interface[permutation]

    # calculate connection vectors
    connection_vectors = coordinates_interface[1:,:] - coordinates_interface[:-1,:]
    
    # calculate tanget vector
    orthogonal_vectors = np.zeros(connection_vectors.shape)
    orthogonal_vectors[:,0] = connection_vectors[:,1]
    orthogonal_vectors[:,1] -= connection_vectors[:,0]
    # normalize
    orthogonal_vectors /= np.repeat(np.linalg.norm(orthogonal_vectors, axis = 1).reshape(-1,1), 3, axis=1)

    n = np.vstack((np.array([0.0, -1.0, 0.0]), orthogonal_vectors))
    n = n[inverse_permutation]
    
    return n
