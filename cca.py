# %%
import numpy as np
from scipy.signal import welch, butter, filtfilt
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt


def generate_continuous_signal(n, fs=1.0, frequencies=[5, 10, 20]):
    """
    Generate a continuous signal with multiple frequencies.

    Parameters:
    n (int): Length of the signal
    fs (float): Sampling frequency
    frequencies (list of float): List of frequencies to include in the signal

    Returns:
    np.ndarray: Continuous signal
    """
    t = np.linspace(0, n / fs, n, endpoint=False)
    continuous_signal = np.zeros_like(t)
    rs = []
    for freq in frequencies:
        r = np.random.uniform(0.2, 0.8)
        rs.append(r)
        continuous_signal += r * np.sin(2 * np.pi * freq * t)
    return continuous_signal, rs


def ideal_filter(signal, lowcut, highcut, fs, order=5):
    """
    Apply an ideal bandpass filter to the signal.

    Parameters:
    signal (np.ndarray): Input signal
    lowcut (float): Low cutoff frequency
    highcut (float): High cutoff frequency
    fs (float): Sampling frequency
    order (int): Order of the filter

    Returns:
    np.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def generate_reference_signals(n, fs, frequencies):
    """
    Generate a list of reference signals with given frequencies.

    Parameters:
    n (int): Length of the signal
    fs (float): Sampling frequency
    frequencies (list of float): List of frequencies for the reference signals

    Returns:
    list of np.ndarray: List of reference signals
    """
    t = np.linspace(0, n / fs, n, endpoint=False)
    reference_signals = []
    for freq in frequencies:
        reference_signals.append(
            (np.cos(2 * np.pi * freq * t), f'cos({freq}Hz)'))
        reference_signals.append(
            (np.sin(2 * np.pi * freq * t), f'sin({freq}Hz)'))
    return reference_signals


def apply_cca(signal, reference_signals, n_components=1):
    """
    Apply Canonical Correlation Analysis (CCA) to the signal.

    Parameters:
    signal (np.ndarray): Input signal
    reference_signals (list of np.ndarray): List of reference signals
    n_components (int): Number of components to keep

    Returns:
    np.ndarray: Transformed signal
    np.ndarray: CCA weights
    """
    cca = CCA(n_components=n_components)
    reference_signals_stacked = np.column_stack(
        [ref_signal for ref_signal, _ in reference_signals])
    signal_transformed, reference_transformed = cca.fit_transform(
        reference_signals_stacked, signal.reshape(-1, 1))
    return signal_transformed.flatten(), cca.x_weights_


def plot_signals(original_signal, filtered_signal, cca_signal):
    """
    Plot the original, filtered, and CCA signals.

    Parameters:
    original_signal (np.ndarray): Original signal
    filtered_signal (np.ndarray): Filtered signal
    cca_signal (np.ndarray): CCA signal
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='Original Signal')
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.plot(cca_signal, label='CCA Signal')
    plt.title("Comparison of Original, Filtered, and CCA Signals")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def plot_reference_signals(reference_signals, cca_weights):
    """
    Plot the reference signals with CCA weights in a separate graph.

    Parameters:
    reference_signals (list of np.ndarray): List of reference signals
    cca_weights (np.ndarray): CCA weights
    """
    plt.figure(figsize=(10, 6))
    for (ref_signal, label), weight in zip(reference_signals, cca_weights.flatten()):
        plt.plot(ref_signal, label=label, alpha=abs(weight), linewidth=2 +
                 3 * abs(weight), color=plt.cm.viridis(abs(weight)))
    plt.title("Reference Signals with CCA Weights")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


# Parameters
n = 200
fs = 1000.0
lowcut = 0.1
highcut = 30.0
signal_frequencies = [5, 10, 20]  # Frequencies for the continuous signal
reference_frequencies = [1, 5, 10, 15, 20]  # Frequencies for reference signals

# Generate continuous signal
original_signal, rs = generate_continuous_signal(n, fs, signal_frequencies)

# Apply ideal filter
filtered_signal = ideal_filter(original_signal, lowcut, highcut, fs)

# Generate reference signals
reference_signals = generate_reference_signals(n, fs, reference_frequencies)

# Apply CCA
cca_signal, cca_weights = apply_cca(original_signal, reference_signals)

# Plot signals
plot_signals(original_signal, filtered_signal, cca_signal)
plot_reference_signals(reference_signals, cca_weights)
print(list(float(e) for e in cca_weights.squeeze()))
print(rs)

# %%
