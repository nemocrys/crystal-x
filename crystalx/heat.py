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
        self._solution = dolfinx.Function(V, name="T")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

    @property
    def solution(self):
        return self._solution

    @property
    def test_function(self):
        return self._test_function
    
    def setup(self, T, dV, dA, dI, rho, kappa, omega, varsigma, h, T_amb, A, f):
        heat_scaling = 2.76661542784358 # Value from Steady state
        
        Form_T = (
            kappa * ufl.inner(ufl.grad(T), ufl.grad(self._test_function))
            - rho * ufl.inner(f, self._test_function)
            - heat_scaling * varsigma / 2 * omega ** 2 * ufl.inner(ufl.inner(A, A), self._test_function)
        ) * 2*pi*self._r*  dV + h * ufl.inner((T("-") - T_amb), self._test_function("-")) * 2*pi*self._r* (
            dI(Surface.crystal.value)
            + dI(Surface.melt.value)
            + dI(Surface.crucible.value)
        )

        # material parameters for radiation
        with open("examples/materials/materials.yml") as f:
            mat_data = yaml.safe_load(f)


        sigma_sb = 5.670374419e-8
        for vol, surf in zip([Volume.crystal, Volume.melt, Volume.crucible, Volume.insulation], [Surface.crystal, Surface.melt, Surface.crucible, Surface.insulation]):
            eps = mat_data[vol.name]["Emissivity"]
            Form_T += (
                sigma_sb
                # * varepsilon("-")
                * eps
                * ufl.inner((T("-") ** 4 - T_amb ** 4), self._test_function("-"))
                * 2*pi*self._r* dI(surf.value)
            )

        # TODO additional heat source for phase boundary
        v_pull = 4  # mm/min
        v_pull *= 1.6666666e-5  # m/s
        latent_heat_value = 5.96e4 * mat_data["melt"]["Density"] * v_pull  # W/m^2 #TODO: WRONG !!! v_growth is needed and is not uniform!!!!
    
        Form_T += (
            ufl.inner(-latent_heat_value, self._test_function("+")) 
            * 2*pi*self._r* dI(Interface.melt_crystal.value)
        )

        return Form_T

    def assemble(self, Form, bcs):
        Gain_T = ufl.derivative(Form,  self._solution,  self._d_T)

        self._problem_T = dolfinx.fem.NonlinearProblem(Form,  self._solution, bcs, J=Gain_T)

    def solve(self):

        solver_T = dolfinx.NewtonSolver(MPI.COMM_WORLD,  self._problem_T)
        # TODO set proper parameters
        solver_T.atol = 1e-8
        solver_T.rtol = 1e-8
        solver_T.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        solver_T.solve(self._solution)

        return self._solution