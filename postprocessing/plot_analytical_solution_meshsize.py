import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv

################################
#          input               #
################################
# simulation data
directories = [
    # variation in mesh size
    "results_air-size=0.8_mesh-size=1_winding-d=0.01",
    "results_air-size=0.8_mesh-size=0.5_winding-d=0.01",
    "results_air-size=0.8_mesh-size=0.25_winding-d=0.01",
]
file_name = "figures/comparison-analytical-numerical_meshsize"
# parameters in simulation for computation of analytical solution
r_e = 0.06  # = crucible radius
l = 0.05  # = crucible height
N = 3  # number of windings in experiment
I = 100  # A
omega = 2 * np.pi* 13.5e3  # similar to experiment
sigma = 5.88e4  # S/m

#############################
# compute and plot analytical solution
rho = 1/sigma
H = N*I/l
delta = (2 * rho / (omega * 4e-7 * np.pi))**0.5
print("delta =", delta)
m = 2**0.5 * r_e / delta
print('m =', m)
print(f"H = {H} A/m")

# Analytical solution according to Lupi2017
def J(xi, H, delta, m):
    J = (-1j)**0.5 * H * 2**0.5 / delta * jv(1, (-1j)**0.5 * m * xi) / jv(0, (-1j)**0.5 * m)
    return J

xi = np.linspace(0, 1, 1000)
J_xi = J(xi, H, delta, m)
w = rho * np.abs(J_xi)**2 / 2

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.grid(linestyle=":")
line, = ax.plot(xi * r_e *1000, w/1e6)
line.set_label('analytical')

################################
# evaluate and plot simulations

deviation_from_analytical = []
mesh_sizes = []

for directory in directories:
    # numerical long coil
    df = pd.read_csv(f'./{directory}/line-data.csv')
    E = omega * (df["imag_A"].to_numpy()**2 + df["real_A"].to_numpy()**2)**0.5
    j = sigma * E
    heating_rms= 1/sigma *(j/2**0.5)**2
    r = df['Points:0'].to_numpy()
    line, = ax.plot(r * 1000, heating_rms/1e6)
    line.set_label(f"$h_{{max}} = {float(directory.split('_')[2].split('=')[1])*10}$ mm")
    print(f"Deviation at r_max {directory}: {(w.max() - heating_rms.max())/ w.max() * 100}%")
    deviation_from_analytical.append((w.max() - heating_rms.max())/ w.max() * 100)
    mesh_sizes.append(float(directory.split("_")[2].split("=")[1]))

ax.legend()
ax.set_xlabel( 'radius [mm]')
ax.set_ylabel('joule heat $\\left[\\frac{\\mathrm{MW}}{\mathrm{m}^3}\\right]$')
fig.tight_layout()
fig.savefig(f"{file_name}.svg")
fig.savefig(f"{file_name}.png")
# plt.show()


fig3, ax3 = plt.subplots(1, 1, figsize=(6, 4))
ax3.grid(linestyle=":")
line, = ax3.plot(mesh_sizes, deviation_from_analytical, "x-")
ax3.set_xlabel("$h_{max}$ ")
ax3.set_ylabel("deviation at $r=60~\\mathrm{mm}$ [%]")
fig3.tight_layout()
fig3.savefig(f"{file_name}_deviation-rmax.svg")
fig3.savefig(f"{file_name}_deviation-rmax.png")
# plt.show()

### zoom ###
r_zoom = 59.9

plt.rcParams.update({'font.size': 14})
fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
ax2.grid(linestyle=":")
radius = np.array(xi * r_e * 1000)
power = np.array(w/1e6)
filter = radius >= r_zoom
line, = ax2.plot(radius[filter], power[filter])
line.set_label('analytical')

for directory in directories:
    # numerical long coil
    df = pd.read_csv(f'./{directory}/line-data.csv')
    E = omega * (df["imag_A"].to_numpy()**2 + df["real_A"].to_numpy()**2)**0.5
    j = sigma * E
    heating_rms= 1/sigma *(j/2**0.5)**2
    r = df['Points:0'].to_numpy()
    radius = r * 1000
    power = heating_rms/1e6
    filter = radius >= r_zoom
    line, = ax2.plot(radius[filter], power[filter])
    line.set_label(f"$h_{{max}} = {float(directory.split('_')[2].split('=')[1])*10}$ mm")

# ax2.legend()
# ax2.set_xlabel( 'radius [mm]')
# ax2.set_ylabel('joule heat $\\left[\\frac{\\mathrm{MW}}{\mathrm{m}^3}\\right]$')
fig2.tight_layout()
fig2.savefig(f"{file_name}_zoom.svg")
fig2.savefig(f"{file_name}_zoom.png")
# plt.show()
