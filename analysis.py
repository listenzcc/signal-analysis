# %%
import numpy as np
from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import welch

# ...existing code...


def generate_cos_signal(n):
    """
    Generate an n-length cosine signal.

    Parameters:
    n (int): Length of the signal

    Returns:
    np.ndarray: n-length cosine signal
    """
    t = np.linspace(0, 1, n, endpoint=False)
    cos_signal = np.cos(80 * np.pi * t)
    return cos_signal


def add_noise(signal, noise_level=0.5, noise_type='gaussian', copy=True):
    """
    Add noise to the signal.

    Parameters:
    signal (np.ndarray): Original signal
    noise_level (float): Standard deviation of the noise
    noise_type (str): Type of noise to add ('gaussian' or 'rayleigh')
    copy (bool): If True, return a copy of the signal with noise added. Otherwise, modify the original signal.

    Returns:
    np.ndarray: Signal with added noise
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, signal.shape)
    elif noise_type == 'rayleigh':
        noise = np.random.rayleigh(noise_level, signal.shape)
    else:
        raise ValueError(
            "Unsupported noise type. Use 'gaussian' or 'rayleigh'.")

    if copy:
        return signal + noise
    else:
        signal += noise
        return signal


def plot_signal(original_signal, noised_signal, noise_type='gaussian'):
    """
    Plot the original and noised signals using seaborn.

    Parameters:
    original_signal (np.ndarray): Original signal
    noised_signal (np.ndarray): Signal with added noise
    noise_type (str): Type of noise added ('gaussian' or 'rayleigh')
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=original_signal, label='Original Signal')
    sns.lineplot(data=noised_signal, label=f'Noised Signal ({noise_type})')
    plt.title(f"Cosine Signal with {noise_type.capitalize()} Noise")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    return plt.gcf()


def compute_snr(original_signal, noised_signal):
    """
    Compute the signal-to-noise ratio (SNR).

    Parameters:
    original_signal (np.ndarray): Original signal
    noised_signal (np.ndarray): Signal with added noise

    Returns:
    float: Signal-to-noise ratio in dB
    """
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean((original_signal - noised_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_snr_with_spectrum(original_signal, noised_signal, fs=1.0):
    """
    Compute the signal-to-noise ratio (SNR) using the spectrum.

    Parameters:
    original_signal (np.ndarray): Original signal
    noised_signal (np.ndarray): Signal with added noise
    fs (float): Sampling frequency

    Returns:
    float: Signal-to-noise ratio in dB
    """
    nperseg = min(1024, len(original_signal))
    f_orig, Pxx_den_orig = welch(original_signal, fs, nperseg=nperseg)
    f_noise, Pxx_den_noise = welch(noised_signal, fs, nperseg=nperseg)

    signal_power = np.sum(Pxx_den_orig)
    noise_power = np.sum(Pxx_den_noise - Pxx_den_orig)

    # Ensure noise_power is not negative or zero
    noise_power = max(noise_power, 1e-10)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def plot_spectrum(original_signal, noised_signal, fs=1.0, noise_type='gaussian'):
    """
    Plot the spectrum of the original and noised signals in a single graph.

    Parameters:
    original_signal (np.ndarray): Original signal
    noised_signal (np.ndarray): Signal with added noise
    fs (float): Sampling frequency
    noise_type (str): Type of noise added ('gaussian' or 'rayleigh')
    """
    nperseg = min(1024, len(original_signal))
    f_orig, Pxx_den_orig = welch(original_signal, fs, nperseg=nperseg)
    f_noise, Pxx_den_noise = welch(noised_signal, fs, nperseg=nperseg)

    plt.figure(figsize=(10, 4))
    plt.semilogy(f_orig, Pxx_den_orig, label='Original Signal')
    plt.semilogy(f_noise, Pxx_den_noise, label=f'Noised Signal ({noise_type})')
    plt.title(f"Signal Spectrum with {noise_type.capitalize()} Noise")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power Spectral Density [V^2/Hz]")
    plt.legend()
    plt.grid()
    return plt.gcf()


def compare_snr_for_noise_levels(noise_levels, signal_length=100, num_trials=10, noise_type='gaussian'):
    """
    Compare SNR with and without the spectrum method for multiple noise levels.

    Parameters:
    noise_levels (list): List of noise levels to compare
    signal_length (int): Length of the signal
    num_trials (int): Number of trials to run for each noise level
    noise_type (str): Type of noise to add ('gaussian' or 'rayleigh')
    """
    original_signal = generate_cos_signal(signal_length)
    snr_values = {level: [] for level in noise_levels}
    snr_spectrum_values = {level: [] for level in noise_levels}

    for noise_level in noise_levels:
        for _ in range(num_trials):
            noised_signal = add_noise(
                original_signal, noise_level=noise_level, noise_type=noise_type)
            snr = compute_snr(original_signal, noised_signal)
            snr_spectrum = compute_snr_with_spectrum(
                original_signal, noised_signal)
            snr_values[noise_level].append(snr)
            snr_spectrum_values[noise_level].append(snr_spectrum)

    snr_means = [np.mean(snr_values[level]) for level in noise_levels]
    snr_spectrum_means = [np.mean(snr_spectrum_values[level])
                          for level in noise_levels]
    snr_stds = [np.std(snr_values[level]) for level in noise_levels]
    snr_spectrum_stds = [np.std(snr_spectrum_values[level])
                         for level in noise_levels]

    plt.figure(figsize=(10, 4))
    plt.errorbar(noise_levels, snr_means, yerr=snr_stds,
                 label=f'SNR ({noise_type})', fmt='-o')
    plt.errorbar(noise_levels, snr_spectrum_means, yerr=snr_spectrum_stds,
                 label=f'SNR using Spectrum ({noise_type})', fmt='-o')
    plt.title(f"SNR Comparison for Different Noise Levels ({
              noise_type.capitalize()})")
    plt.xlabel("Noise Level (Standard Deviation)")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid()
    return plt.gca()


# %%
original_signal = generate_cos_signal(8000)
noised_signal_gaussian = add_noise(original_signal, noise_type='gaussian')
noised_signal_rayleigh = add_noise(original_signal, noise_type='rayleigh')

fig_gaussian = plot_signal(
    original_signal, noised_signal_gaussian, noise_type='gaussian')
fig_rayleigh = plot_signal(
    original_signal, noised_signal_rayleigh, noise_type='rayleigh')

fig_spectrum_gaussian = plot_spectrum(
    original_signal, noised_signal_gaussian, noise_type='gaussian')
fig_spectrum_rayleigh = plot_spectrum(
    original_signal, noised_signal_rayleigh, noise_type='rayleigh')

snr_gaussian = compute_snr(original_signal, noised_signal_gaussian)
print(
    f"Signal-to-Noise Ratio (SNR) with Gaussian Noise: {snr_gaussian:.2f} dB")

snr_rayleigh = compute_snr(original_signal, noised_signal_rayleigh)
print(
    f"Signal-to-Noise Ratio (SNR) with Rayleigh Noise: {snr_rayleigh:.2f} dB")

snr_spectrum_gaussian = compute_snr_with_spectrum(
    original_signal, noised_signal_gaussian)
print(
    f"Signal-to-Noise Ratio (SNR) using Spectrum with Gaussian Noise: {snr_spectrum_gaussian:.2f} dB")

snr_spectrum_rayleigh = compute_snr_with_spectrum(
    original_signal, noised_signal_rayleigh)
print(
    f"Signal-to-Noise Ratio (SNR) using Spectrum with Rayleigh Noise: {snr_spectrum_rayleigh:.2f} dB")

# Compare SNR for different noise levels
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
fig_compare_gaussian = compare_snr_for_noise_levels(
    noise_levels, noise_type='gaussian')
fig_compare_rayleigh = compare_snr_for_noise_levels(
    noise_levels, noise_type='rayleigh')
plt.show()

# %%
# ...existing code...
