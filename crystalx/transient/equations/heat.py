import ufl
import dolfinx
from mpi4py import MPI

import yaml

from numpy import pi

from geometry.geometry import Volume, Interface, Surface, Boundary

class Heat:
    def __init__(self, V) -> None:
        # trial function 
        self._d_T = ufl.TrialFunction(V)
        # test function
        self._test_function = ufl.TestFunction(V)
        # solution variable
        self._solution = dolfinx.fem.Function(V, name="T")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

        self._heat_scaling = 2.6552669163700564 #2.6567555396547373 # Value from Steady state

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function
    
    def setup(self, T, T_old, Dt, dV, dA, dI, rho, capacity, kappa, omega, varsigma, h, T_amb, A, f):
        
        Form_T = 1/Dt * ufl.inner(capacity * rho * (T-T_old), self._test_function) * 2*pi*self._r*  dV
        Form_T += (
            kappa * ufl.inner(ufl.grad(T), ufl.grad(self._test_function))
            - rho * ufl.inner(f, self._test_function)
            - self._heat_scaling * varsigma / 2 * omega ** 2 * ufl.inner(ufl.inner(A, A), self._test_function)
        ) * 2*pi*self._r*  dV
        
        Form_T += h * ufl.inner((T("-") - T_amb), self._test_function("-")) * 2*pi*self._r* (
            dI(Surface.crystal.value)
            + dI(Surface.meniscus.value)
            + dI(Surface.melt_flat.value)
            + dI(Surface.crucible.value)
        )

        # material parameters for radiation
        with open("examples/materials/materials.yml") as f:
            mat_data = yaml.safe_load(f)


        sigma_sb = 5.670374419e-8
        for vol, surf in zip([Volume.axis_top, Volume.seed ,Volume.crystal, Volume.melt, Volume.melt, Volume.crucible, Volume.insulation, Volume.adapter, Volume.axis_bottom, Volume.inductor], [Surface.axis_top, Surface.seed, Surface.crystal, Surface.meniscus,Surface.melt_flat, Surface.crucible, Surface.insulation, Surface.adapter, Surface.axis_bottom, Surface.inductor]):
            eps = mat_data[vol.material]["Emissivity"]
            Form_T += (
                sigma_sb
                # * varepsilon("-")
                * eps
                * ufl.inner((T("-") ** 4 - T_amb ** 4), self._test_function("-"))
                * 2*pi*self._r* dI(surf.value)
            )

        return Form_T

    def assemble(self, Form, bcs):
        Gain_T = ufl.derivative(Form,  self._solution,  self._d_T)

        self._problem_T = dolfinx.fem.petsc.NonlinearProblem(Form,  self._solution, bcs, J=Gain_T)

    def solve(self):

        solver_T = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD,  self._problem_T)
        # TODO set proper parameters
        solver_T.atol = 1e-8
        solver_T.rtol = 1e-8
        solver_T.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        # from dolfinx.log import set_log_level, LogLevel
        # set_log_level(LogLevel.INFO)

        n, converged = solver_T.solve(self._solution)
        self._solution.x.scatter_forward()

        assert(converged)
        # print(f"Number of interations: {n:d}")

        return self._solution