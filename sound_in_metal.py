import numpy as np
import pandas as pd

from Constants import BRASS_LENGTH, ALUMINIUM_LENGTH, COPPER_LENGTH, MEASUREMENT_UNCERTAINTY_METAL_TIME, METAL_LENGTH_UNCERTENTY

#column names
measurement_column = "Time per 5 periods (s)"
corrected_meshurment = "Time per period (s)"
relative_error = "relative error"
absolute_error = "error"

def read_data(metal: str):
    df = pd.read_csv("sound_in_metal_data/" + metal + "_data.csv")
    df[corrected_meshurment] = df[measurement_column] /5
    return df

def average_period(metal: str):
    df = read_data(metal)
    return df[corrected_meshurment].mean()

def speed_of_sound(metal: str) -> float:
    period = average_period(metal)
    if metal == "Aluminium":
        length = ALUMINIUM_LENGTH
    elif metal == "Brass":
        length = BRASS_LENGTH
    elif metal == "Copper":
        length = COPPER_LENGTH
    return (2 * length) / period, length

def read_data_with_error_type_B(metal: str):
    df = read_data(metal)
    df[relative_error] = MEASUREMENT_UNCERTAINTY_METAL_TIME / df[measurement_column] 
    df[absolute_error] = df[corrected_meshurment] * df[relative_error]
    return df

def average_period_error(metal: str):
    df = read_data(metal)
    N = len(df)
    
    # 1. Type A Uncertainty: Standard Error of the Mean
    # ddof=1 ensures we use the sample standard deviation (denominator N-1)
    standard_deviation = df[corrected_meshurment].std(ddof=1)
    uncertainty_type_A = standard_deviation / np.sqrt(N)
    
    # 2. Type B Uncertainty: Systematic device uncertainty
    # Since it's a systematic offset, taking an average does not reduce this error.
    uncertainty_type_B = MEASUREMENT_UNCERTAINTY_METAL_TIME / 5
    
    # 3. Total Uncertainty: Combined in quadrature
    total_period_uncertainty = np.sqrt(uncertainty_type_A**2 + uncertainty_type_B**2)
    return total_period_uncertainty

def speed_of_sound_error(metal: str):
    v, length = speed_of_sound(metal)
    period = average_period(metal)
    delta_period = average_period_error(metal)
    delta_length = METAL_LENGTH_UNCERTENTY
    
    # Using Gaussian error propagation for v = 2 * length / period
    # Relative error of v is the Pythagorean sum of relative errors of length and period
    rel_error_length = delta_length / length
    rel_error_period = delta_period / period
    
    rel_error_speed = np.sqrt(rel_error_length**2 + rel_error_period**2)
    
    # Convert back to absolute error
    delta_speed = v * rel_error_speed
    return delta_speed



def main(metal: str):
    print(f"--- Data Sheet for {metal} ---")
    print(read_data_with_error_type_B(metal))
    print("\n--- Final Evaluations ---")
    print(f"Average Period: {average_period(metal)} +- {average_period_error(metal)} s")
    print(f"Speed of sound: {speed_of_sound(metal)[0]} +- {speed_of_sound_error(metal)} m/s")

main("Aluminium")
print(" ")
print(" ")
main("Copper")
print(" ")
print(" ")
main("Brass")




