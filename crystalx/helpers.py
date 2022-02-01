from math import modf

import dolfinx
import ufl

def timeToStr(totalTime):
    millis, sec = modf(totalTime)
    millis = round(millis, 3) * 1000
    sec, millis = int(sec), int(millis)
    if sec < 1:
        return f"{millis}ms"
    elif sec < 60.0:
        return f"{sec}s {millis}ms"
    else:
        return f"{sec // 60}min {(sec % 60)}s {millis}ms"

def project(function, functionspace, **kwargs):
    """
    Computes the L2-projection of the function into the functionspace.
    See: https://fenicsproject.org/qa/6832/what-is-the-difference-between-interpolate-and-project/
    """
    if "name" in kwargs.keys():
        assert isinstance(kwargs["name"], str)
        sol = dolfinx.Function(functionspace, name = kwargs["name"])
    else:
        sol = dolfinx.Function(functionspace)
    
    w = ufl.TrialFunction(functionspace)
    v = ufl.TestFunction(functionspace)

    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(function, v) * ufl.dx

    problem = dolfinx.fem.LinearProblem(a, L, bcs=[])
    sol.interpolate(problem.solve())

    return sol