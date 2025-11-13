"""
distribucion_binomial.py
Cálculo y visualización de la distribución binomial.
Muestra comparaciones manuales vs scipy.stats y grafica la PMF.
Al ejecutar, se mostrarán las figuras (no se guardan en disco).
Dependencias: numpy, scipy, matplotlib, seaborn
"""
import numpy as np
from math import comb
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def manual_binomial_pmf(n, p, k):
    return comb(n, k) * (p**k) * ((1-p)**(n-k))

def main():
    # Parámetros
    n = 50
    p = 0.02
    k = 2

    # Cálculo manual
    pmf_manual = manual_binomial_pmf(n, p, k)

    # Cálculo con scipy
    rv = stats.binom(n, p)
    pmf_scipy = rv.pmf(k)

    print(f"Binomial(n={n}, p={p}), k={k})")
    print(f"PMF manual = {pmf_manual:.6f}")
    print(f"PMF scipy  = {pmf_scipy:.6f}")

    # Graficar PMF completa para k desde 0 hasta n (o hasta 15 para claridad)
    ks = np.arange(0, 16)
    pmf_vals = rv.pmf(ks)

    plt.figure(figsize=(9,5))
    sns.barplot(x=ks, y=pmf_vals, color='tab:blue')
    plt.title(f"PMF Binomial (n={n}, p={p})")
    plt.xlabel("k")
    plt.ylabel("P(X=k)")
    plt.show()

if __name__ == "__main__":
    main()
