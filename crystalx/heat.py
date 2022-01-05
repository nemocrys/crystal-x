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
        self._del_T = ufl.TestFunction(V)
        # solution variable
        self._T = dolfinx.Function(V, name="T")

        # radial coordinate
        self._r = ufl.SpatialCoordinate(V.mesh)[0]

    def assemble_system(self, T, dV, dA, dI, rho, kappa, omega, varsigma, h, T_amb, A, f):
        heat_scaling = 1.0
        
        Form_T = (
            kappa * ufl.inner(ufl.grad(T), ufl.grad(self._del_T))
            - rho * ufl.inner(f, self._del_T)
            - heat_scaling * varsigma / 2 * omega ** 2 * ufl.inner(ufl.inner(A, A), self._del_T)
        ) * 2*pi*self._r*  dV + h * ufl.inner((T("-") - T_amb), self._del_T("-")) * 2*pi*self._r* (
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
                * ufl.inner((T("-") ** 4 - T_amb ** 4), self._del_T("-"))
                * 2*pi*self._r* dI(surf.value)
            )

        # TODO additional heat source for phase boundary
        v_pull = 4  # mm/min
        v_pull *= 1.6666666e-5  # m/s
        latent_heat_value = 5.96e4 * mat_data["crystal"]["Density"] * v_pull  # W/m^2
    
        Form_T += (
            ufl.inner(-latent_heat_value, self._del_T("+")) 
            * 2*pi*self._r* dI(Interface.melt_crystal.value)
        )

        return Form_T

    def setup(self, dV, dA, dI, rho, kappa, omega, varsigma, h, T_amb, A, f, bcs):
        Form_T = self.assemble_system(self._T, dV, dA, dI, rho, kappa, omega, varsigma, h, T_amb, A ,f)
        # TODO what is Re(A), Im(A)
        Gain_T = ufl.derivative(Form_T,  self._T,  self._d_T)

        self._problem_T = dolfinx.fem.NonlinearProblem(Form_T,  self._T, bcs, J=Gain_T)

    def solve(self):

        solver_T = dolfinx.NewtonSolver(MPI.COMM_WORLD,  self._problem_T)
        # TODO set proper parameters
        solver_T.atol = 1e-8
        solver_T.rtol = 1e-8
        solver_T.convergence_criterion = "incremental"
        # parameters copied from https://jorgensd.github.io/dolfinx-tutorial/chapter2/hyperelasticity.html

        solver_T.solve(self._T)

        return self._T