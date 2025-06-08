import numpy as np
import matplotlib.pyplot as plt

# Define particle and grain sizes in mm
dp_mm = 0.02  # particle diameter in mm (20 microns)
dg_mm = 0.6   # grain diameter in mm (600 microns)

# Define depth in mm and h = dg_mm
depth_mm = np.linspace(0, 700, 500)
h = dg_mm
depth_d = depth_mm / h  # normalize depth by h

# Compute r' using Hall's formula: r' = 35 * c * (dp/dg)^(3/2) with c = 0.1
c = 0.1
r_prime = 35 * c * (dp_mm / dg_mm) ** (3 / 2)

# Compute accumulated sediment fraction
P = 1 - np.exp(-r_prime * depth_d)

# Plot
plt.figure(figsize=(7, 6))
plt.plot(P, depth_mm, label=f'dp={dp_mm} mm, dg={dg_mm} mm, c={c}')
plt.xlabel("Accumulated Sediment (fraction of total)")
plt.ylabel("Depth from Surface (mm)")
plt.title("Accumulated Sediment vs Depth (mm units, c=0.1)")
plt.grid(True)
plt.xlim(0, 1.0)
plt.ylim(700, 0)
plt.legend()
plt.tight_layout()
plt.show()
