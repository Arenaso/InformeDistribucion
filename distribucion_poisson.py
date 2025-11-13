"""
distribucion_poisson.py
Cálculo y visualización de la distribución de Poisson.
Muestra comparaciones manuales vs scipy.stats y grafica la PMF.
Dependencias: numpy, scipy, matplotlib, seaborn
"""
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def manual_poisson_pmf(lam, k):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def main():
    lam = 4
    k = 6

    pmf_manual = manual_poisson_pmf(lam, k)
    rv = stats.poisson(mu=lam)
    pmf_scipy = rv.pmf(k)

    print(f"Poisson(lambda={lam}), k={k}")
    print(f"PMF manual = {pmf_manual:.6f}")
    print(f"PMF scipy  = {pmf_scipy:.6f}")

    ks = np.arange(0, 13)
    pmf_vals = rv.pmf(ks)

    plt.figure(figsize=(9,5))
    sns.barplot(x=ks, y=pmf_vals, color='tab:green')
    plt.title(f"PMF Poisson (lambda={lam})")
    plt.xlabel("k")
    plt.ylabel("P(X=k)")
    plt.show()

if __name__ == "__main__":
    main()
