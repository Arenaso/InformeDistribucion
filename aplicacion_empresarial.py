"""
aplicacion_empresarial.py
Simulación Monte Carlo de demanda diaria modelada por Poisson.
Muestra probabilidad de stockout, ingresos y ganancias promedio.
Al ejecutar, se imprimen métricas y se muestran histogramas.
Dependencias: numpy, scipy, matplotlib, seaborn, pandas
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")

def simulate_demand(lam, stock, price, cost, N=10000, random_seed=42):
    np.random.seed(random_seed)
    demands = np.random.poisson(lam, size=N)
    sales = np.minimum(demands, stock)
    revenues = price * sales
    profits = (price - cost) * sales
    stockout_prob = np.mean(demands > stock)
    return {
        "demands": demands,
        "sales": sales,
        "revenues": revenues,
        "profits": profits,
        "stockout_prob": stockout_prob
    }

def main():
    # Parámetros (ejemplo)
    lam = 20.5  # demanda media diaria estimada
    stock = 25
    price = 10.0
    cost = 6.0
    N = 20000

    results = simulate_demand(lam, stock, price, cost, N=N)

    print(f"Estimated stockout probability: {results['stockout_prob']:.4f}")
    print(f"Average daily revenue: ${results['revenues'].mean():.2f}")
    print(f"Average daily profit: ${results['profits'].mean():.2f}")

    # Mostrar dataframe resumen
    df = pd.DataFrame({
        "demand": results["demands"],
        "sales": results["sales"],
        "revenue": results["revenues"],
        "profit": results["profits"]
    })
    summary = df.describe()
    print("\\nSummary statistics:")
    print(summary)

    # Histograma de demandas simuladas
    plt.figure(figsize=(10,5))
    sns.histplot(results["demands"], bins=30, kde=False)
    plt.title("Histogram of simulated daily demands (Poisson)")
    plt.xlabel("Demand")
    plt.ylabel("Frequency")
    plt.show()

    # Boxplot de profits
    plt.figure(figsize=(8,4))
    sns.boxplot(x=results["profits"])
    plt.title("Distribution of daily profits (simulated)")
    plt.xlabel("Profit")
    plt.show()

if __name__ == "__main__":
    main()
