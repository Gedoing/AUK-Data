import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from odrpack import odr_fit

from Constants import RESONANCE_LENGTH_UNCERTENTY, FREQUENCY_UNCERTAINTY

def read_data():
    df = pd.read_csv("resonance_in_air.csv")
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# FIX: odrpack models expect x first, then beta: f(x, beta)
# beta[0] = slope (m), beta[1] = intercept (c)
def linear_model(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] * x + beta[1]


def analyze_and_graph(df: pd.DataFrame, frequency: int):
    row = df[df["Frequency (Hz)"] == frequency]
    if row.empty:
        raise ValueError(f"Frequency {frequency} Hz not found in dataset.")

    lengths_series = row.drop(columns=["Frequency (Hz)"]).squeeze()
    lengths_series = pd.to_numeric(lengths_series).dropna()

    node_numbers = (
        lengths_series.index.str.extract(r"(\d+)")[0].astype(int).to_numpy()
    )
    values = lengths_series.to_numpy()

    # Define initial guesses for the parameter vector [slope, intercept]
    slope_guess = (values[-1] - values[0]) / (
        node_numbers[-1] - node_numbers[0]
    )
    beta0 = [slope_guess, values[0]]

    # ODRPACK expects weights (1 / sigma^2) instead of standard deviations
    weight_y = np.full_like(values, 1.0 / (RESONANCE_LENGTH_UNCERTENTY**2))

    # -------------------------------------------------------------------------
    # Corrected execution via standalone odrpack interface
    # -------------------------------------------------------------------------
    sol = odr_fit(
        f=linear_model,
        xdata=node_numbers,
        ydata=values,
        beta0=beta0,
        weight_y=weight_y,  # <-- FIX: replaced sy with weight_y
        task="explicit-ODR",
    )

    m, c = sol.beta  # Extracted optimal fit parameters
    sigma_m, sigma_c = sol.sd_beta  # Extracted standard errors

    # Calculate final physical speed of sound: v = 2 * m * f
    v = 2 * m * frequency

    # Rigorous total differential to account for parameter coupling
    sigma_v = (2 * frequency * sigma_m) + (2 * m * FREQUENCY_UNCERTAINTY)

    print(f"\n--- Results for f = {frequency} Hz ---")
    print(f"Slope m:        ({m:.5f} ± {sigma_m:.5f}) m")
    print(f"Intercept c:    ({c:.5f} ± {sigma_c:.5f}) m")
    print(f"Speed of Sound: ({v:.2f} ± {sigma_v:.2f}) m/s")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)

    # Raw measurement points
    ax.errorbar(
        node_numbers,
        values,
        yerr=RESONANCE_LENGTH_UNCERTENTY,
        fmt="o",
        linestyle="None",
        color="darkblue",
        ecolor="darkred",
        capsize=3,
        label=f"Data ($f = {frequency}$ Hz)",
    )

    # Continuous ODR regression path
    x_fit = np.linspace(node_numbers.min() - 0.2, node_numbers.max() + 0.2, 100)
    y_fit = linear_model(x_fit, [m, c])
    ax.plot(
        x_fit,
        y_fit,
        color="black",
        linestyle="--",
        label=f"ODR Fit ($v = {v:.1f}\\pm{sigma_v:.1f}$ m/s)",
    )

    ax.set_xlabel("Resonance Node Number ($n$)")
    ax.set_ylabel("Resonance Position $x_n$ (m)")
    ax.set_title(f"Resonance Curve at $f = {frequency}$ Hz")
    ax.set_xticks(np.arange(node_numbers.min(), node_numbers.max() + 1))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    return fig



def main():
    plt.close("all")
    df = read_data()

    # Map across your active frequencies
    frequenzen = [600, 1000, 1500] 
    for f in frequenzen:
        analyze_and_graph(df, f)

    plt.show()

main()