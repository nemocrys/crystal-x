import ufl
import dolfinx
from mpi4py import MPI

from numpy import pi

class Laplace:
    def __init__(self, V) -> None:
        # trial function 
        self._d_V = ufl.TrialFunction(V)
        # test function
        self._test_function = ufl.TestFunction(V)
        # solution variable
        self._solution = dolfinx.fem.Function(V, name="V")

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function
    
    def setup(self, V, dV, dA, dI):
        Form = (
            ufl.inner(ufl.grad(V), ufl.grad(self._test_function))
        ) * dV

        return Form

    def assemble(self, Form, bcs):
        Gain = ufl.derivative(Form,  self._solution,  self._d_V)

        self._problem = dolfinx.fem.petsc.NonlinearProblem(Form,  self._solution, bcs, J=Gain)

    def solve(self):

        solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD,  self._problem)
        # TODO set proper parameters
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        solver.solve(self._solution)

        return self._solution