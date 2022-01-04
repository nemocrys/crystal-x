import ufl
import dolfinx
from mpi4py import MPI

from numpy import pi

from geometry.geometry import Volume, Interface, Surface, Boundary

def solve(V, dV, dA, dI, mu, omega, varsigma, current_density, bcs):
    d_A = ufl.TrialFunction(V)
    del_A = ufl.TestFunction(V)
    A = dolfinx.Function(V, name="A")
    
    r = ufl.SpatialCoordinate(V.mesh)[0]

    # Option 1: write Form directly
    Form_A = (
        1
        / mu
        * (
            # ufl.inner(ufl.rot(A), ufl.rot(del_A))
            ufl.inner(1/r*(r*A).dx(0), 1/r*(r*del_A).dx(0))
            + ufl.inner(-A.dx(1), -del_A.dx(1))
            # ufl.inner(A.dx(i), del_A.dx(i))
        )
        + 1j * omega * varsigma * ufl.inner(A, del_A)
    ) * 2*pi*r *dV - ufl.inner(current_density, del_A) * 2*pi*r * dV(Volume.inductor.value)

    # TODO what is Re(A), Im(A)
    Gain_A = ufl.derivative(Form_A, A, d_A)

    problem_EM = dolfinx.fem.NonlinearProblem(Form_A, A, bcs, J=Gain_A)
    solver_EM = dolfinx.NewtonSolver(MPI.COMM_WORLD, problem_EM)
    # TODO set proper parameters
    solver_EM.atol = 1e-8
    solver_EM.rtol = 1e-8
    solver_EM.convergence_criterion = "incremental"
    # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

    solver_EM.solve(A)

    return A