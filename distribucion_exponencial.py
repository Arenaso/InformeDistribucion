"""
distribucion_exponencial.py
Cálculo y visualización de la distribución exponencial.
Compara cálculo manual de P(X > t) vs scipy.stats survival function.
Dependencias: numpy, scipy, matplotlib, seaborn
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="whitegrid")

def main():
    # Media = 10 horas => lambda = 1/10
    lam = 1/10
    t = 8.0

    # Manual P(X > t) = exp(-lambda * t)
    p_manual = math.exp(-lam * t)

    # Scipy uses scale = 1/lambda
    rv = stats.expon(scale=1/lam)
    p_scipy = rv.sf(t)  # survival function P(X>t)

    print(f"Exponencial(lambda={lam}), P(X > {t})")
    print(f"P manual = {p_manual:.6f}")
    print(f"P scipy  = {p_scipy:.6f}")

    # Graficar PDF
    xs = np.linspace(0, 40, 400)
    pdf_vals = lam * np.exp(-lam * xs)

    plt.figure(figsize=(9,5))
    plt.plot(xs, pdf_vals, lw=2)
    plt.title(f"PDF Exponencial (lambda={lam})")
    plt.xlabel("x (horas)")
    plt.ylabel("f(x)")
    plt.show()

if __name__ == "__main__":
    main()
