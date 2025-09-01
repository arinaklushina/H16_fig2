import numpy as np
import scipy as sc
import os
import matplotlib.pyplot as plt


def S_S_0(r):
    return a / (r + a)

M = 10**11 * 1.989e33       # 10^11 solar masses in g
gamma_cm = 8.5243e-30       # cm^-1
k_cm = 9.54e-54             # cm^-2
c = 2.99792458e10           # cm/s
G = 6.67430e-8              # cm^3 g^-1 s^-1
kpc_to_cm = 3.0857e21       # 1 kpc in cm

r_g_kpc = (G * M / c**2) / kpc_to_cm
beta_kpc = r_g_kpc
gamma_kpc = gamma_cm * kpc_to_cm
w = 1 - 6 * beta_kpc * gamma_kpc # in kpc
a = (1+w)/gamma_kpc
k_kpc = k_cm * kpc_to_cm**2

r_kpc = np.linspace(1, 10**3, 1000)

#finding h (40)

# Parameters
beta_val = float(beta_kpc)
w_val = float(w)
gamma_val = float(gamma_kpc)
k_val = float(k_kpc)

# Polynomial: -k r^3 + gamma r^2 + w r - 2 beta = 0
coeffs = [-k_val, gamma_val, w_val, -2.0 * beta_val]

# Find roots
roots = np.roots(coeffs)

# Filter positive, nearly real roots
tol_imag = 1e-10
real_positive_roots = [
    float(r.real) for r in roots if abs(r.imag) < tol_imag and r.real > 0
]

# Get smallest positive root
h = min(real_positive_roots) if real_positive_roots else None

print("h =", h)

import sympy as sp

# --- Physical parameters ---
M = 10 ** 11 * 1.989e33  # g
gamma_cm = 8.5243e-30  # cm⁻¹
k_cm = 9.54e-54  # cm⁻²
c = 2.99792458e10  # cm/s
G = 6.67430e-8  # cm³ g⁻¹ s⁻²
kpc_to_cm = 3.0857e21  # 1 kpc = 3.0857e21 cm

# Convert to kpc units
r_g_kpc = (G * M / c ** 2) / kpc_to_cm
beta_kpc_val = float(r_g_kpc)  # Ensure float
gamma_kpc_val = float(gamma_cm * kpc_to_cm)  # in kpc⁻¹
k_kpc_val = float(k_cm * (kpc_to_cm) ** 2)  # in kpc⁻²

# Derived values
w_val = 1 - 6 * beta_kpc_val * gamma_kpc_val  # float
a_val = (1 + w_val) / gamma_kpc_val

r_1_val = 1 / gamma_kpc_val  # in kpc
r_2_val = 1 / np.sqrt(k_kpc_val)  # in kpc

# Radius array
r_kpc = np.linspace(1, 1000, 1000)  # 1 to 1000 kpc


# --- Define S/S₀ ---
def S_S_0(r):
    return a_val / (r + a_val)


# --- Symbolic B(r) ---
r_sym = sp.symbols('r', real=True, positive=True)
w_sym, beta_sym, r1_sym, r2_sym = sp.symbols('w beta r1 r2', real=True)

B_expr = w_sym - 2 * beta_sym / r_sym + r_sym / r1_sym - (r_sym / r2_sym) ** 2
B_prime_expr = sp.diff(B_expr, r_sym)

print("B(r) =", B_expr)
print("B'(r) =", B_prime_expr)

# --- Lambdify: Convert to numerical function ---
# Use 'numpy' mode so it accepts arrays
B_func = sp.lambdify((r_sym, w_sym, beta_sym, r1_sym, r2_sym), B_expr, 'numpy')

# --- Evaluate B(r) numerically ---
try:
    B_vals = B_func(
        r_kpc,  # NumPy array
        w_val,  # float
        beta_kpc_val,  # float
        r_1_val,  # float
        r_2_val  # float
    )
except Exception as e:
    print("Lambdify error:", e)
    raise

# --- Ensure B_vals is a NumPy array of floats (not symbolic) ---
if not isinstance(B_vals, np.ndarray):
    B_vals = np.array(B_vals, dtype=float)

# --- Compute B_tilde(r) = [S/S₀(r_g)]² * B(r) ---

S_vals = S_S_0(r_kpc)
B_tilde_vals = (S_vals) ** 2 * B_vals

# --- Mask where B_tilde_vals > 0 ---
# Now B_tilde_vals should be a NumPy array
valid = B_tilde_vals > 0

x_vals = np.log10(S_S_0(r_kpc[valid]) * r_kpc[valid])

# --- x = gamma / (2k) in kpc ---
x_special_1 = gamma_kpc / (2 * k_kpc)
x_special_2 = (2 * beta_kpc_val / gamma_kpc) ** (1 / 2)
x_special_3 = (beta_kpc_val / k_kpc) ** (1 / 3)

if not np.any(valid):
    print("❌ No valid points where B_tilde > 0")
    print("B_tilde range:", B_tilde_vals.min(), "to", B_tilde_vals.max())
else:
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(x_vals, np.log10(B_tilde_vals[valid]),
            label=fr'$\tilde{{B}}(r)$: $\beta={beta_kpc_val:.2e}, \gamma={gamma_kpc_val:.2e}, k={k_kpc_val:.2e}$',
            color='darkblue', linewidth=2)
    # Vertical dashed line at x_special_1
    ax.axvline(np.log10(x_special_1), color='green', linestyle='--',
               label=fr'$r = \frac{{\gamma}}{{2 \kappa}} \approx {x_special_1:.4g}\ \mathrm{{kpc}}$')

    # Vertical dashed line at x_special_2
    ax.axvline(np.log10(x_special_2), color='yellow', linestyle='--',
               label=fr'$r = \sqrt{{\frac{{2 \beta}}{{\gamma}}}} \approx {x_special_2:.4g}\ \mathrm{{kpc}}$')

    # Vertical dashed line at x_special_3
    ax.axvline(np.log10(x_special_3), color='black', linestyle='--',
               label=fr'$r = \left(\frac{{\beta}}{{\kappa}}\right)^{{1/3}} \approx {x_special_3:.4g}\ \mathrm{{kpc}}$')

    ax.set_xlabel(r'$\log_{10}((S/S_0)(r) \times r)$')
    ax.set_ylabel(r'$\log_{10}(\tilde{{B}}(r))$')
    ax.set_title(r'$\tilde{B}(r)$ vs $r$')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)

    # --- Save ---
    fig.savefig("B_tilde_plot_HF.png", dpi=200, bbox_inches='tight')
    print("✅ Plot saved as 'B_tilde_plot_HF.png'")

    plt.show()

