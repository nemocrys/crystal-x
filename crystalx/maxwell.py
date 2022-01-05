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
        self._del_A = ufl.TestFunction(V)
        # solution variable
        self._A = dolfinx.Function(V, name="A")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

    def assemble_system(self, A, dV, dA, dI, mu, omega, varsigma, current_density):
        Form_A = (
            1
            / mu
            * (
                # ufl.inner(ufl.rot(A), ufl.rot(del_A))
                ufl.inner(1 / self._r * ( self._r *  A).dx(0), 1/ self._r * (self._r *  self._del_A).dx(0))
                + ufl.inner(-A.dx(1), -self._del_A.dx(1))
                # ufl.inner(A.dx(i), del_A.dx(i))
            )
            + 1j * omega * varsigma * ufl.inner(A,  self._del_A)
        ) * 2*pi* self._r *dV - ufl.inner(current_density,  self._del_A) * 2*pi* self._r * dV(Volume.inductor.value)

        return Form_A

    def setup(self, dV, dA, dI, mu, omega, varsigma, current_density, bcs):
        Form_A = self.assemble_system(self._A, dV, dA, dI, mu, omega, varsigma, current_density)
        # TODO what is Re(A), Im(A)
        Gain_A = ufl.derivative(Form_A,  self._A,  self._d_A)

        self._problem_EM = dolfinx.fem.NonlinearProblem(Form_A,  self._A, bcs, J=Gain_A)

    def solve(self):

        solver_EM = dolfinx.NewtonSolver(MPI.COMM_WORLD,  self._problem_EM)
        # TODO set proper parameters
        solver_EM.atol = 1e-8
        solver_EM.rtol = 1e-8
        solver_EM.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        solver_EM.solve(self._A)

        return self._A