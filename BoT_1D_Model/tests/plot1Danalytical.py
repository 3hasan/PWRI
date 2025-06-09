import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytical1Dsolution import fetter_analytical_solution

# Testing if the analytical solution works as intended using a dummy 1D grid and dummy values
L = np.linspace(0, 5, 500)      
v_x = 3.0                       
t = 1.0                         
C_injection = 20.0             
D_L = 0.01                      

C_values = np.array([fetter_analytical_solution(x, t, C_injection, v_x, D_L) for x in L])

plt.figure(figsize=(8, 5))
plt.plot(L, C_values, label=f"t = {t} s")
plt.xlabel("Distance $L$ (cm)")
plt.ylabel("Concentration $C$ (ppm)")
plt.title("Analytical Step-Change Solution at t = 1 s")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
