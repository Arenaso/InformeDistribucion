"""
distribucion_normal.py
Cálculo y visualización de la distribución normal.
Compara cálculo manual usando la función de error (erf) vs scipy.stats CDF.
Dependencias: numpy, scipy, matplotlib, seaborn, mpmath (optional)
"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(style="whitegrid")

def manual_normal_cdf(x, mu, sigma):
    # CDF usando la función error (erf)
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def main():
    mu = 30.0
    sigma = 5.0
    x = 25.0

    cdf_manual = manual_normal_cdf(x, mu, sigma)
    rv = stats.norm(loc=mu, scale=sigma)
    cdf_scipy = rv.cdf(x)

    print(f"Normal(mu={mu}, sigma={sigma}), P(X < {x})")
    print(f"CDF manual = {cdf_manual:.6f}")
    print(f"CDF scipy  = {cdf_scipy:.6f}")

    # Graficar la PDF
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf_vals = rv.pdf(xs)

    plt.figure(figsize=(9,5))
    plt.plot(xs, pdf_vals, lw=2)
    plt.axvline(x, color='red', linestyle='--', label=f"x={x}")
    plt.title(f"PDF Normal (mu={mu}, sigma={sigma})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
