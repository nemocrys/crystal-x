import ufl
import dolfinx
from mpi4py import MPI

from numpy import pi

from geometry.geometry import Volume, Interface, Surface, Boundary

class Maxwell:
    def __init__(self, V) -> None:
        # trial function 
        self._d_A = ufl.TrialFunction(V)
        # test function
        self._test_function = ufl.TestFunction(V)
        # solution variable
        self._solution = dolfinx.fem.Function(V, name="A")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function

    def setup(self, A, dV, dA, dI, mu, omega, varsigma, current_density):
        Form_A = (
            1
            / mu
            * (
                # ufl.inner(ufl.rot(A), ufl.rot(del_A))
                ufl.inner(1 / self._r * ( self._r *  A).dx(0), 1/ self._r * (self._r *  self._test_function).dx(0))
                + ufl.inner(-A.dx(1), -self._test_function.dx(1))
                # ufl.inner(A.dx(i), del_A.dx(i))
            )
            + 1j * omega * varsigma * ufl.inner(A,  self._test_function)
        ) * 2*pi* self._r *dV - ufl.inner(current_density,  self._test_function) * 2*pi* self._r * dV(Volume.inductor.value)

        return Form_A

    def assemble(self, Form, bcs):
        
        # TODO what is Re(A), Im(A)
        Gain_A = ufl.derivative(Form,  self._solution,  self._d_A)

        self._problem_EM = dolfinx.fem.petsc.NonlinearProblem(Form,  self._solution, bcs, J=Gain_A)

    def solve(self):

        solver_EM = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD,  self._problem_EM)
        # TODO set proper parameters
        solver_EM.atol = 1e-8
        solver_EM.rtol = 1e-8
        solver_EM.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        solver_EM.solve(self._solution)

        return self._solution