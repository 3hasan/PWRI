import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot dP vs PVI using some experimental data and Carmen-Kozeny formulation to check for the sanity 

#  Load & clean data
df = pd.read_csv('pv_dp.csv', sep=';')
df.columns = ['PVI', 'dp_mbar']
df['PVI']     = df['PVI'].astype(float)
df['dp_mbar'] = df['dp_mbar'].astype(float)

# Parameters
# There is a core shoulder (hence _sh) of smaller diameter with length 1.5 mm and there is the _main core
Q_ml_min = 5.0
Q         = Q_ml_min*1e-6/60.0    # m3 /s
C_ppmw    = 16.0                  # mg solids per L
rho_w     = 1e3                   # kg/m3
rho_s     = 2.642e3               # kg/m3
mu        = 0.89e-3               # Pa·s

L_sh    = 1.5e-3                  # m
A_sh    = np.pi*(20.9e-3/2)**2
L_main  = 50e-3                   # m
A_main  = 5.23e-4
L_tot   = L_sh + L_main

phi0    = 0.24
Vp_tot  = phi0*(A_sh*L_sh + A_main*L_main)


k0      = 1.9 * 9.869e-13 # initial permeability m2
ck0     = phi0**3/(1-phi0)**2 # CK fraction multiplier
# Permeability of the invaded zone
def k_dep(phi):
    return k0 * ((phi**3/(1-phi)**2) / ck0)

# ΔP at single depth & PVI
def model_dp_at_depth(L_dep, pvi):
    L_dep_sh = min(L_dep,   L_sh)
    L_dep_m  = max(0.0,     L_dep - L_sh)
    L_clean  = L_tot - L_dep
    
    Vp_dep   = phi0*(A_sh*L_dep_sh + A_main*L_dep_m)
    Vw       = pvi * Vp_tot
    m_s      = Vw * rho_w * (C_ppmw*1e-6)
    V_s      = m_s / rho_s
    
    phi_dep  = max(phi0 - V_s/Vp_dep, 1e-6)
    kc       = k_dep(phi_dep)
    
    dp1 = mu * Q * L_dep_sh / (kc   * A_sh)
    dp2 = mu * Q * L_dep_m  / (kc   * A_main)
    dp3 = mu * Q * L_clean  / (k0    * A_main)
    
    return (dp1 + dp2 + dp3)/100.0   # → mbar

# Best-fit penetration by bisection between 1.5 and 10 mm
pvi_last    = df['PVI'].iloc[-1]
dp_exp_last = df['dp_mbar'].iloc[-1]

a, b    = 1.5e-3, 5e-3
fa, fb  = model_dp_at_depth(a, pvi_last) - dp_exp_last, model_dp_at_depth(b, pvi_last) - dp_exp_last

for _ in range(30):
    m    = 0.5*(a + b)
    fm   = model_dp_at_depth(m, pvi_last) - dp_exp_last
    if fa*fm <= 0:
        b, fb = m, fm
    else:
        a, fa = m, fm

L_opt = 0.5*(a + b)
print(f"Best-fit penetration depth ≈ {L_opt*1e3:.2f} mm "
      f"(model ΔP={model_dp_at_depth(L_opt, pvi_last):.2f} mbar, "
      f"exp ΔP={dp_exp_last:.2f} mbar)")

# Plotted deposit depths + best fit to match the last data point
deposit_mm = [0.5, 1.5, 2, 3, 5, 6, 7, 8, 9, L_opt*1e3]
deposit_m  = [d*1e-3 for d in deposit_mm]

model_curves = {}
for Lm, Lm_m in zip(deposit_mm, deposit_m):
    model_curves[Lm] = [ model_dp_at_depth(Lm_m, pvi) for pvi in df['PVI'] ]

# Plot
plt.figure(figsize=(8,5))
plt.plot(df['PVI'], df['dp_mbar'], 'ko', label='Experiment')
for Lm, curve in model_curves.items():
    style = '-' if abs(Lm - L_opt*1e3)<1e-3 else '--'
    plt.plot(df['PVI'], curve, style,
             label=f'{Lm:.1f} mm{" (best)" if abs(Lm - L_opt*1e3)<1e-3 else ""}')

plt.xlabel('Pore Volumes Injected (–)')
plt.ylabel('ΔP [mbar]')
plt.title('ΔP vs. PVI: Exp., CK‐Model & Best‐Fit Depth')
plt.legend(fontsize='small', ncol=2)
plt.ylim(0, df['dp_mbar'].max()*1.5)
plt.grid(True)
plt.tight_layout()
plt.show()
