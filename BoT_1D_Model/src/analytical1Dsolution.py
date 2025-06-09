import numpy as np
from scipy.special import erfc, erfcx

'Analytical Solution Function'

# from  Fetter et al. (2018), Contaminant Hydrogeology, p.77-78 2.9.3 One-Dimensional Step Change in Concentration(First-Type Boundary)
# C= C_injection/2 * [erfc((L-v_x*t)/(2*sqrt(D_L*t)) + exp (v_x*L/D_L) * erfc((L+v_x*t)/(2*sqrt(D_L_t))] = C_inj/2 * erfc(minus_term) + exp (v_x*L/D_L) * erfc(plus_term)

# Notice erfc(plus term) is scaled with erfcx to avoid overflow, so there is a -plus_term**2 in the exponential. 
# erfcx(x) = exp(x**2) * erfc(x)

def fetter_analytical_solution(L, t, C_injection, v_x, D_L):
    if t <= 0:
        return 0.0

    minus_term = (L - v_x * t) / (2 * np.sqrt(D_L * t))
    plus_term = (L + v_x * t) / (2 * np.sqrt(D_L * t))

    erfc_minus = erfc(minus_term)
    scaled_second_term = np.exp((v_x * L) / D_L - plus_term**2) * erfcx(plus_term)
    
    return 0.5 * C_injection * (erfc_minus + scaled_second_term)