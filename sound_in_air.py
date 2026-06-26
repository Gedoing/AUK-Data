import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import odrpack

from matplotlib.ticker import FuncFormatter

from Constants import SOUNT_IN_AIR_TIME_UNCERTENTY, SOUNT_IN_AIR_DISTANCE_UNCERTENTY, MOLAR_MASS_AIR, IDEAL_GAS_CONSTSNT, ROOM_THEMPERATURE, ROOM_THEMPERATURE_UNCERTENTY


#column names
average = "Average time"
ablosut_error_time = "Error"
uncertenty_type_A_time = "uncertenty type A"
uncertainty_type_B_time = "uncertainty type B"

def get_mean():
    df = pd.read_csv("sound_in_air_data.csv")
    columns_to_average = ["Time 1 (s)", "Time 2 (s)", "Time 3 (s)"]
    df[average] = df[columns_to_average].mean(axis=1)

    N = len(columns_to_average)
    df[uncertenty_type_A_time] = df[columns_to_average].std(axis=1, ddof=1) / np.sqrt(N)
    df[uncertainty_type_B_time] = SOUNT_IN_AIR_TIME_UNCERTENTY
    df[ablosut_error_time] = np.sqrt(df[uncertenty_type_A_time]**2 + df[uncertainty_type_B_time]**2)
    return df

# CRITICAL CHANGE: In odrpack, the independent variable x must be the FIRST argument
def linear_model_func(x, beta):
    """y = m * x + c -> beta[0] is slope (v), beta[1] is intercept"""
    return beta[0] * x + beta[1]


def fit_and_plot_sound_speed(df: pd.DataFrame):
    # 1. Extract variables (x = Time, y = Distance)
    x_data = df["Average time"].to_numpy()
    y_data = df["Distance (m)"].to_numpy()

    # Absolute uncertainties
    x_err = df["Error"].to_numpy()
    y_err = np.full_like(y_data, SOUNT_IN_AIR_DISTANCE_UNCERTENTY)

    # 2. Set up odrpack functional arguments
    # odrpack uses weights: weight = 1 / (uncertainty^2)
    weight_x = 1.0 / (x_err**2)
    weight_y = 1.0 / (y_err**2)

    # Initial guess: slope ~ 340 m/s, intercept ~ 0.0
    beta0 = np.array([340.0, 0.0])

    # Run the fit directly using the clean functional interface
    fit_result = odrpack.odr_fit(linear_model_func, x_data, y_data, beta0, weight_x=weight_x, weight_y=weight_y)

    # Extract fit parameters from the fit_result object
    v_sound = fit_result.beta[0]
    v_sound_error = fit_result.sd_beta[0]
    intercept = fit_result.beta[1]

    # 3. Plotting
    fig = plt.figure(figsize=(8, 5), dpi=100)

    # Plot data points with error bars on both axes
    plt.errorbar(
        x_data,
        y_data,
        xerr=x_err,
        yerr=y_err,
        fmt="o",
        color="black",
        ecolor="darkred",
        capsize=3,
        label="Data points (with uncertainties)",
    )

    # Generate points for the fit line
    x_fit = np.linspace(x_data.min() * 0.9, x_data.max() * 1.1, 100)
    y_fit = linear_model_func(x_fit, fit_result.beta)

    plt.plot(
        x_fit,
        y_fit,
        color="blue",
        linestyle="--",
        label=f"ODR Fit: $v$ = {v_sound:.1f} ± {v_sound_error:.1f} m/s",
    )

    #change x axis label to ms
    def to_ms(x, pos):
        return f"{x * 1000:.2f}"
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_ms))

    # Labels and formatting
    plt.xlabel("Propagation Time $t$ (ms)")
    plt.ylabel("Distance $d$ (m)")
    plt.title("Determination of the Speed of Sound in Air")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    return v_sound, v_sound_error, intercept, fig


def calculate_adiabatic_index(velocety: float, uncertenty_velosety: float):
    v = velocety
    delta_v = uncertenty_velosety
    M = MOLAR_MASS_AIR
    T = ROOM_THEMPERATURE
    R = IDEAL_GAS_CONSTSNT
    k = (v**2 * M) / (R * T)

    relative_delta_v = delta_v / v
    relative_delta_T = ROOM_THEMPERATURE_UNCERTENTY / ROOM_THEMPERATURE

    delta_k = k * np.sqrt((2 * relative_delta_v) ** 2 + (relative_delta_T) ** 2)

    return k, delta_k

def main():
    v_sound,v_sound_error, intercept, fig = fit_and_plot_sound_speed(get_mean())
    adiabatic_index, adiabatic_index_error = calculate_adiabatic_index(v_sound, v_sound_error)

    print(get_mean()[["Distance (m)", "Time 1 (s)", "Time 2 (s)", "Time 3 (s)", average, ablosut_error_time]])
    print(f"Calculated Speed of Sound: {v_sound:.2f} +/- {v_sound_error:.2f} m/s")
    print(f"Intercept at: {intercept}")
    print(f"The adiabatic index is : {adiabatic_index} +/- {adiabatic_index_error}")
    fig.savefig("Figures/sound in air")
    plt.show()

main()


