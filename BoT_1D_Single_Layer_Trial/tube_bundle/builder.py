import numpy as np
import pandas as pd
from pathlib import Path


def load_throat_distribution(path, sep=';'):
    """
    Load throat-size distribution from a CSV.

    Supports two formats:
      - Cumulative percent: 'diameter_mm', 'cum_pct'
      - PDF: 'radius_micron', 'freq'
    Default sep=';' for semicolon-delimited files.
    Returns DataFrame with trimmed column names.
    """
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


def sample_tube_radii(df, N, seed=None):
    """
    Sample N unscaled tube radii (m) from the distribution.

    - If 'cum_pct' & 'diameter_mm' present: use CDF of diameters (mm).
    - If 'radius_micron' & 'freq' present: use PDF of radii (µm).
    """
    rng = np.random.default_rng(seed)

    if 'cum_pct' in df.columns and 'diameter_mm' in df.columns:
        diam_m = df['diameter_mm'].values * 1e-3
        cdf = np.clip(df['cum_pct'].values / 100.0, 0, 1)
        us = rng.random(N)
        sampled_d = np.interp(us, cdf, diam_m)
        return sampled_d / 2.0  # to radii (m)

    if 'radius_micron' in df.columns and 'freq' in df.columns:
        radii_um = df['radius_micron'].values
        probs = df['freq'].values
        probs = probs / probs.sum()
        idx = rng.choice(len(radii_um), size=N, p=probs)
        return radii_um[idx] * 1e-6  # to meters

    raise ValueError("DataFrame must have either cum_pct & diameter_mm or radius_micron & freq.")


def compute_moments(df):
    """
    Compute exact moments M2 and M4 from distribution:
      - If CDF style: uses 'diameter_mm' & 'cum_pct'.
      - If PDF style: uses 'radius_micron' & 'freq'.
    Returns:
      M2 = E[r^2], M4 = E[r^4] with r in meters.
    """
    # Cumulative-percent format
    if 'diameter_mm' in df.columns and 'cum_pct' in df.columns:
        arr = df[['diameter_mm', 'cum_pct']].dropna().sort_values('diameter_mm')
        d = arr['diameter_mm'].to_numpy() * 1e-3  # m
        cdf = np.clip(arr['cum_pct'].to_numpy() / 100.0, 0, 1)
        # PDF from CDF increments
        pdf = np.diff(np.concatenate(([0.0], cdf)))
        r = d / 2.0
        M2 = np.sum(pdf * r**2)
        M4 = np.sum(pdf * r**4)
        return M2, M4
    
    # PDF format
    if 'radius_micron' in df.columns and 'freq' in df.columns:
        r = df['radius_micron'].to_numpy() * 1e-6  # convert µm to m
        pdf = df['freq'].to_numpy().astype(float)
        pdf = pdf / pdf.sum()
        M2 = np.sum(pdf * r**2)
        M4 = np.sum(pdf * r**4)
        return M2, M4

    raise ValueError("DataFrame must contain either CDF or PDF columns for moments calculation.")


def compute_optimal_N_alpha(phi, area, K, M2, M4):
    """
    Solve for number of tubes N and scale factor alpha
    to exactly match porosity and permeability.

    Equations:
      alpha^2 * N * pi * M2 = phi * area
      alpha^4 * N * pi * M4 = 8 * area * K
    """
    # Correct two-equation solution:
    # From combining equations, N = (phi^2 * A * M4) / (8 * K * pi * M2^2)
    N_exact = (phi**2 * area * M4) / (8 * K * np.pi * M2**2)
    N = int(np.ceil(N_exact))

    # Compute alpha from porosity equation: alpha^2 = (phi * area) / (N * pi * M2)
    alpha = np.sqrt((phi * area) / (N * np.pi * M2))
    return N, alpha


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
    # Analytical moments (no sampling error)
    M2, M4 = compute_moments(df)
    N, alpha = compute_optimal_N_alpha(phi, area, K, M2, M4)
    # Sample radii (use a different seed offset if provided)
    radii_unscaled = sample_tube_radii(df, N, seed)
    radii_scaled = radii_unscaled * alpha
    return radii_scaled, N, alpha


if __name__ == "__main__":
    # Project-root–relative path resolution
    base = Path(__file__).parent.parent
    data_file = base / 'data' / 'throat_pdf.csv'

    # Model parameters
    PHI     = 0.246                  # porosity
    A       = 5.23e-4               # cross-sectional area (m^2)
    K_mD    = 1850                  # permeability in mD
    K       = K_mD * 9.869e-16      # convert to m^2
    SEED    = 42                    # reproducible RNG seed

    # Build tube bundle 
    radii, N_tubes, alpha = build_tube_bundle(
        data_file, PHI, A, K,
        sep=';', seed=SEED
    )

    # Results
    area_sum = np.sum(np.pi * radii**2)
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
