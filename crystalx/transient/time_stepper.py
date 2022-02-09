import ufl

class OneStepTheta:
    def __init__(self, problem, theta=0) -> None:
        assert theta >= 0 and theta <=1

        self._problem = problem
        self._theta = theta

    def step(self, old_solution, scaling, t, Dt, dV, *kwargs):
        old_F = self._problem.setup(old_solution, dV, *kwargs)
        new_F = self._problem.setup(self._problem.solution, dV, *kwargs)

        Form = scaling * ufl.inner(self._problem.solution - old_solution, self._problem.test_function) * dV + Dt * ( self._theta * new_F - (1 - self._theta) * old_F )

        return Form