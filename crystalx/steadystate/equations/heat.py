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

        self._heat_scaling = 0.0

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function
    
    def setup(self, T, dV, dA, dI, rho, kappa, omega, varsigma, varepsilon, v_pull, T_amb, A, f, material_data):
        
        Form_T = (
            kappa * ufl.inner(ufl.grad(T), ufl.grad(self._test_function))
            - rho * ufl.inner(f, self._test_function)
            - self._heat_scaling * varsigma / 2 * omega ** 2 * ufl.inner(ufl.inner(A, A), self._test_function)
        ) * 2*pi*self._r*  dV 
        
        for vol, surf in zip([Volume.crystal, Volume.melt, Volume.melt, Volume.crucible], [Surface.crystal, Surface.meniscus,Surface.melt_flat, Surface.crucible]):
            h = material_data[vol.material]["Heat Transfer"]
            Form_T += (
                h
                * ufl.inner((T("-")  - T_amb ), self._test_function("-"))
                * 2*pi*self._r* dI(surf.value)
            )

        sigma_sb = 5.670374419e-8
        for vol, surf in zip([Volume.axis_top, Volume.seed ,Volume.crystal, Volume.melt, Volume.melt, Volume.crucible, Volume.insulation, Volume.adapter, Volume.axis_bottom, Volume.inductor], [Surface.axis_top, Surface.seed, Surface.crystal, Surface.meniscus,Surface.melt_flat, Surface.crucible, Surface.insulation, Surface.adapter, Surface.axis_bottom, Surface.inductor]):
            eps = material_data[vol.material]["Emissivity"]
            Form_T += (
                sigma_sb
                # * varepsilon("-") # It is important to choose the right side of the interface
                * eps
                * ufl.inner((T("-") ** 4 - T_amb ** 4), self._test_function("-"))
                * 2*pi*self._r* dI(surf.value)
            )

        # Additional heat source for phase boundary
        latent_heat_value = material_data["tin-solid"]["Latent Heat"] * material_data["tin-liquid"]["Density"] * v_pull
    
        Form_T += (
            ufl.inner(-latent_heat_value, ufl.avg(self._test_function)) 
            * 2*pi*self._r* dI(Interface.melt_crystal.value)
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

        n, converged = solver_T.solve(self._solution)
        self._solution.x.scatter_forward()

        assert(converged)

        return self._solution