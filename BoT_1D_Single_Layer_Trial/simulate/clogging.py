import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read from the tube radii CSV
root = Path(__file__).parent.parent
radii = pd.read_csv(root/'data'/'tube_radii.csv')['radius_m'].to_numpy()

# Read from the particle diameter CSV
parts    = pd.read_csv(root/'data'/'particle_cum_pct.csv', sep=';')
parts    = parts.sort_values('diameter_micron')
diam_um  = parts['diameter_micron'].values
cdf      = parts['cum_pct'].values / 100.0

def fraction_above(r_m):
    d_um = r_m * 2 * 1e6 / 1.25
    cum  = np.interp(d_um, diam_um, cdf, left=0.0, right=1.0)
    return 1 - cum

# Parameters
A_m2   = 5.23e-4        # core cross‐section [m2]
phi    = 0.246          # porosity
L_t    = 1.5e-3         # inlet segment length [m]
L_r    = 51.5e-3        # remaining core length [m]
L_tot  = L_t + L_r

Q      = 5.0e-6/60      # m^3/s (5 mL/min)
C_s    = 16e-6          # mass fraction
rho_s  = 2.642e6        # g/m^3
m_s    = C_s * Q * 1e6  # g/s solids

# average particle volume from CDF
r_um   = diam_um / 2
pdf    = np.diff(np.concatenate(([0.], cdf)))
vols   = 4/3 * np.pi * (r_um*1e-6)**3
V_avg  = np.sum(pdf * vols)

dt    = 3600.0
steps = 48
times = np.arange(steps) * dt  # seconds

K_tube     = np.zeros(steps)
open_tubes = np.ones_like(radii, dtype=bool)

K0_tube    = np.sum(np.pi * radii**4) / (8 * A_m2)
K_tube[0]  = K0_tube

K_rem0     = K0_tube # remaining core perm (constant)

# Clogging loop
for i in range(1, steps):
    # update inlet perm
    K_tube[i] = np.sum(np.pi * radii[open_tubes]**4) / (8 * A_m2)

    # particles arriving this dt
    m_dt  = m_s * dt
    V_dt  = m_dt / rho_s
    N_tot = V_dt / V_avg

    if N_tot <= 0 or not open_tubes.any():
        print(f"t={times[i]:.0f}s: no injection or all tubes closed")
        continue

    # flow bias among open tubes
    r4         = radii**4
    flow_frac  = r4 / np.sum(r4[open_tubes])
    N_j        = N_tot * flow_frac

    # straining fraction and clogging probability
    frac_str   = np.array([fraction_above(r) for r in radii])
    p_clog     = 1 - np.exp(-N_j * frac_str)

    # random closures
    rnd        = np.random.random(size=radii.size)
    new_clog   = (rnd < p_clog) & open_tubes
    n_closed   = new_clog.sum()
    open_tubes[new_clog] = False

    print(f"t={times[i]:.0f}s: closed {n_closed} tubes, K_tube={K_tube[i]:.3e}")

# Harmonic Averaged full core perm
K_full = L_tot / (L_t/K_tube + L_r/K_rem0)

# PVI
pore_vol    = A_m2 * L_tot * phi     # total pore volume (m^3)
V_inj       = Q * times              # cumulative injected volume (m^3)
PVI         = V_inj / pore_vol      # pore‐volumes injected

# PVI vs K
print("\nPVI         K_tube (m^2)      K_full (m^2)")
print("-------------------------------------------------")
for pv, Kt, Kf in zip(PVI, K_tube, K_full):
    print(f"{pv:8.3f}    {Kt:.12e}    {Kf:.12e}")

# Plot
plt.figure(figsize=(6,4))
plt.plot(PVI,     K_tube, '-o', label='Inlet (1.5 mm)')
plt.plot(PVI,     K_full, '-s', label='Full core (53 mm)')
plt.xlabel('Pore Volumes Injected (–)')
plt.ylabel('Permeability $K$ (m$^2$)')
plt.title('Permeability Decline vs. Pore Volumes Injected')
plt.legend()
plt.grid(ls='--', alpha=0.6)
plt.tight_layout()
plt.show()
