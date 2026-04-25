import numpy as np
from scipy.optimize import curve_fit

# Define Gaussian function
def gaussian(x, A, x0, sigma):
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

# Load data from file
filename = r"C:\Users\Владимир\Desktop\NMR_Diplom\final_data\25.00K_f_Bloc.txt"
data = np.loadtxt(filename, skiprows=3)  # Skip header rows

x = data[:, 0]
y = data[:, 1]

# Filter out very small y values to focus on the main peak
mask = y > 1e-10  # Remove near-zero values that might interfere with fitting
x_filtered = x[mask]
y_filtered = y[mask]

# Initial parameter guesses based on the data
A_guess = np.max(y_filtered)  # Peak height
x0_guess = x_filtered[np.argmax(y_filtered)]  # Peak center
sigma_guess = (np.max(x_filtered) - np.min(x_filtered)) / 10  # Rough estimate of width
initial_guess = [A_guess, x0_guess, sigma_guess]

# Perform the fit
popt, pcov = curve_fit(gaussian, x_filtered, y_filtered, p0=initial_guess, maxfev=10000)

A_fit, x0_fit, sigma_fit = popt

print(f"Fitted Gaussian parameters:")
print(f"A (amplitude): {A_fit:.6f}")
print(f"x0 (center): {x0_fit:.6f}")
print(f"sigma (width): {sigma_fit:.6f}")

# Calculate R-squared for goodness of fit
y_pred = gaussian(x_filtered, *popt)
ss_res = np.sum((y_filtered - y_pred) ** 2)
ss_tot = np.sum((y_filtered - np.mean(y_filtered)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.6f}")