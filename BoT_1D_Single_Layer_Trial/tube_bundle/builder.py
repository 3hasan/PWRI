import numpy as np
import pandas as pd
from pathlib import Path


# Read data from the CSV file and create the pandas data frame. The CSV shouldn't have any empty cells.
def load_throat_distribution(path, sep=';'):

    df = pd.read_csv(path, sep=sep)
    return df

# From the dataframe, calculate second moment M2 to match the Porosity and fourth moment M4 to match the Permeability
def compute_moments(df):
    # CDF format
    if 'diameter_mm' in df.columns and 'cum_pct' in df.columns:
        arr = df[['diameter_mm', 'cum_pct']].dropna().sort_values('diameter_mm')
        d = arr['diameter_mm'].to_numpy() * 1e-3  # convert mm to m
        cdf = np.clip(arr['cum_pct'].to_numpy() / 100.0, 0, 1)
        # PDF from CDF increments
        pdf = np.diff(np.concatenate(([0.0], cdf)))
        r = d / 2.0
        M2 = np.sum(pdf * r**2)
        M4 = np.sum(pdf * r**4)
        return M2, M4
    
    # PDF format
    if 'radius_micron' in df.columns and 'freq' in df.columns:
        r = df['radius_micron'].to_numpy() * 1e-6  # convert microns to m
        pdf = df['freq'].to_numpy().astype(float)
        pdf = pdf / pdf.sum()
        M2 = np.sum(pdf * r**2)
        M4 = np.sum(pdf * r**4)
        return M2, M4

    raise ValueError("DataFrame must contain either CDF or PDF columns for moments calculation.")

# Using M2 and M4 solve the two equations for the two unknowns N and scaling factor alpha
# alpha^2 * N * pi * M2 = phi * area
# alpha^4 * N * pi * M4 = 8 * area * K 
# Giving a solution N = (phi^2 * A * M4) / (8 * K * pi * M2^2) 
def compute_optimal_N_alpha(phi, area, K, M2, M4):

    N_exact = (phi**2 * area * M4) / (8 * K * np.pi * M2**2)
    N = int(np.ceil(N_exact))
    alpha = np.sqrt((phi * area) / (N * np.pi * M2))
    return N, alpha

def sample_tube_radii(df, N, seed=None):
    rng = np.random.default_rng(seed)

    # If CDF folder type
    if 'cum_pct' in df.columns and 'diameter_mm' in df.columns:
        diam_m = df['diameter_mm'].values * 1e-3
        cdf = np.clip(df['cum_pct'].values / 100.0, 0, 1)
        us = rng.random(N)
        sampled_d = np.interp(us, cdf, diam_m)
        return sampled_d / 2.0  # to radii (m)

    # If PDF folder type
    if 'radius_micron' in df.columns and 'freq' in df.columns:
        radii_um = df['radius_micron'].values
        probs = df['freq'].values
        probs = probs / probs.sum()
        idx = rng.choice(len(radii_um), size=N, p=probs)
        return radii_um[idx] * 1e-6  # to meters

    raise ValueError("DataFrame must have either cum_pct & diameter_mm or radius_micron & freq.")

def build_tube_bundle(path_csv, phi, area, K, sep=';', seed=None):
    """
    Full pipeline:
      1. Load distribution.
      2. Compute exact M2, M4 from CDF.
      3. Solve for N, alpha.
      4. Sample final N radii and scale by alpha.

    Returns:
      - radii_scaled: numpy array of tube radii (m)
      - N: number of tubes used
      - alpha: scaling factor applied
    """
    df = load_throat_distribution(path_csv, sep)
    M2, M4 = compute_moments(df)
    N, alpha = compute_optimal_N_alpha(phi, area, K, M2, M4)
    radii_unscaled = sample_tube_radii(df, N, seed)
    radii_scaled = radii_unscaled * alpha
    return radii_scaled, N, alpha


if __name__ == "__main__":
    # Project-root–relative path resolution
    base = Path(__file__).parent.parent
    data_file = base / 'data' / 'throat_pdf.csv'

    # Model parameters 
    PHI     = 0.246                  # porosity
    A       = 5.23e-4               # cross-sectional area (m2)
    K_mD    = 1850                  # permeability in miliDarcy
    K       = K_mD * 9.869e-16      # convert to m2
    SEED    = 42                    # reproducible RNG seed to give same sampling each run

    # Build tube bundle 
    radii, N_tubes, alpha = build_tube_bundle(
        data_file, PHI, A, K,
        sep=';', seed=SEED
    )

    # Results
    area_sum = np.sum(np.pi * radii**2) # Total area of the tubes
    # Darcy’s law: Q = K * A/μ * ΔP/L
    # Poiseuille for N parallel tubes: Q = (ΔP / (8 * μ * L)) * sum_j (pi * r_j**4) = K * A/μ * ΔP/L
    # K = sum_j (pi * r_j**4) / 8A
    eff_K    = np.sum(np.pi * radii**4) / (8 * A)


    print(f"Final tubes: {N_tubes}")
    print(f"Scale factor alpha: {alpha:.4f}")
    print(f"Sum pore area:    {area_sum:.2e} m^2 (target {PHI*A:.2e})")
    print(f"Effective K:      {eff_K:.2e} m^2 (target {K:.2e})")

    # Save radii distribution to CSV
    out_file = base / 'data' / 'tube_radii.csv'
    import pandas as _pd
    _df = _pd.DataFrame({'radius_m': radii})
    _df.to_csv(out_file, index=False)
    print(f"Saved tube radii distribution to {out_file}")

