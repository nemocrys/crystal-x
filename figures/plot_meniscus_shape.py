import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 14})

gamma = 0.56
rho = 6980.0
g = 9.81
beta = 10 /360*2*np.pi


l_c = (gamma / (rho * g) )**0.5
h =  l_c*(2 - 2* np.sin(beta))**0.5
z = np.linspace(0, h, 100)[1:]
x = l_c * (
    np.arccosh(2 * l_c/z) - np.arccosh(2*l_c/h)
) - l_c * (
    (4 - z**2/l_c**2)**0.5 - (4 - h**2/l_c**2)**0.5
)
fig, ax = plt.subplots(1, 1)
plt.plot(x, z, "k")
ax.set_ylim([0, 0.004])
ax.axis("equal")
ax.grid(linestyle=":")
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
fig.tight_layout()
fig.savefig("figures/meniscus_shape_transparent.png", transparent=True, dpi=600)
fig.savefig("figures/meniscus_shape.png", dpi=600)
plt.show()
