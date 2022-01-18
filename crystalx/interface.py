from typing import Type
from geometry.geometry import Interface
import ufl
import dolfinx
from mpi4py import MPI

from numpy import pi
from scipy.linalg import solve

class Stefan:
    def __init__(self, V) -> None:
        # trial function 
        self._d_V = ufl.TrialFunction(V)
        # test function
        self._test_function = ufl.TestFunction(V)
        # solution variable
        self._solution = dolfinx.Function(V, name="V_normal")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function
    
    def setup(self, V, dV, dA, dI, kappa, latent_heat, Dt, T):
        N = ufl.FacetNormal(V.function_space.mesh)
        a = (
            ufl.inner(ufl.avg(self._d_V), ufl.avg(self.test_function)) 
        ) * 2*pi*self._r*  dI(Interface.melt_crystal.value) # Need to use trial function here for bilinear form

        L = (Dt / latent_heat * 
            ufl.inner(ufl.jump(ufl.inner(kappa * ufl.grad(T), N)), ufl.avg(self.test_function)) 
        ) * 2*pi*self._r*  dI(Interface.melt_crystal.value)
        
        return dolfinx.fem.form.Form(a), dolfinx.fem.form.Form(L)

    def assemble(self, a, L, bcs, interface_dofs):
        A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
        A.assemble()
        b = dolfinx.fem.create_vector(L)
        dolfinx.fem.assemble_vector(b, L)

        self._A = A[interface_dofs, interface_dofs]
        self._b = b[interface_dofs]
    
    def solve(self, interface_dofs):
        
        with self.solution.vector.localForm() as loc:
            values = loc.getArray()
            values[interface_dofs] = solve(self._A, self._b)
            loc.setArray(values)

        return self._solution