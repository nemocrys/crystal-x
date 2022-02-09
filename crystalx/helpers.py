from math import modf

import dolfinx
import ufl
from petsc4py import PETSc
import numpy as np

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

def save_mesh(mesh, cell_tags, facet_tags, comm, name="mesh", dir=""):
    with dolfinx.io.XDMFFile(comm, f"{dir}{name}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(cell_tags)
        xdmf.write_meshtags(facet_tags)

def load_mesh(comm, name):
    with dolfinx.io.XDMFFile(comm, name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(mesh, name="Cell tags")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
        facet_tags = xdmf.read_meshtags(mesh, name="Facet tags")

    return mesh, cell_tags, facet_tags

def save_function(function, name="function", directory=""):
    coordinates = function.function_space.tabulate_dof_coordinates()
    with function.vector.localForm() as loc:
        values = loc.getArray()
        
        output_array = np.empty(shape=(coordinates.shape[0], coordinates.shape[1] + loc.getBlockSize()), dtype=np.complex128)
        
        for block in range(loc.getBlockSize()):
            output_array[:,block] = values[block::loc.getBlockSize()]
        
        output_array[:, loc.getBlockSize():] = coordinates
        np.savetxt(f"{directory}{name}.txt", output_array.view(float))

# Load in style
# fx1 fy1 fz1 x1 y1 z1
# fx2 fy2 fz2 x2 y2 z2
# ...
# fxn fyn fzn xn yn zn

# https://fenicsproject.discourse.group/t/i-o-from-xdmf-hdf5-files-in-dolfin-x/3122/10
def load_function(V, datafile, normalize=False):
    
    f = dolfinx.Function(V)
    # block size of vector
    bs = f.vector.getBlockSize()
    
    # load data and coordinates
    data = np.loadtxt(datafile,usecols=(np.arange(0,2 * bs))).view(np.complex128)
    coords = np.loadtxt(datafile,usecols=np.arange(2 * -3, 0)).view(np.complex128).real # last three always are the coordinates

    # new node coordinates (dofs might be re-ordered in parallel)
    # in case of DG fields, these are the Gauss point coordinates
    co = V.tabulate_dof_coordinates()

    # index map
    im = V.dofmap.index_map.global_indices()

    tol = 1.0e-8
    tolerance = int(-np.log10(tol))

    # since in parallel, the ordering of the dof ids might change, so we have to find the
    # mapping between original and new id via the coordinates
    ci = 0
    for i in im:
        
        ind = np.where((np.round(coords,tolerance) == np.round(co[ci],tolerance)).all(axis=1))[0]
        
        # only write if we've found the index
        if len(ind):
            
            if normalize:
                norm_sq = 0.
                for j in range(bs):
                    norm_sq += data[ind[0],j]**2.
                norm = np.sqrt(norm_sq)
            else:
                norm = 1.
            
            for j in range(bs):
                f.vector[bs*i+j] = data[ind[0],j] / norm
        
        ci+=1

    f.vector.assemble()
    
    # update ghosts
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    
    return f