# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ...existing code...


def plot_spectrum_leak(signal, sampling_rate):
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Plot using seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=fft_freq, y=np.abs(fft_result))
    plt.title('Spectrum Leak')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    st.pyplot(plt)


def plot_comparison(signal1, signal2, sampling_rate):
    # Compute the FFT of both signals
    fft_result1 = np.fft.fft(signal1)
    fft_freq1 = np.fft.fftfreq(len(signal1), d=1/sampling_rate)

    fft_result2 = np.fft.fft(signal2)
    fft_freq2 = np.fft.fftfreq(len(signal2), d=1/sampling_rate)

    # Plot using seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    sns.lineplot(x=fft_freq1, y=np.abs(fft_result1))
    plt.title('Spectrum with Leaking')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    sns.lineplot(x=fft_freq2, y=np.abs(fft_result2))
    plt.title('Spectrum without Leaking')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    st.pyplot(plt)


def main():
    st.title("Spectrum Leak Analysis")

    st.markdown("""
    ## Introduction
    This application demonstrates the phenomenon of spectrum leakage in signal processing.
    Spectrum leakage occurs when the signal being analyzed is not periodic within the observation window.
    Use the sliders below to adjust the length of the signal with and without leakage and observe the differences in their frequency spectra.
    """)

    sampling_rate = 1000  # 1000 Hz

    length_with_leak = st.slider(
        'Length with Leak', min_value=1, max_value=1000, value=530)
    length_without_leak = st.slider(
        'Length without Leak', min_value=1, max_value=1000, value=1000)

    t = np.linspace(0, 1, sampling_rate, endpoint=False)

    # Signal with leaking (not an integer number of periods)
    signal_with_leak = np.sin(2 * np.pi * 50 * t[:length_with_leak])

    # Signal without leaking (integer number of periods)
    signal_without_leak = np.sin(2 * np.pi * 50 * t[:length_without_leak])

    plot_comparison(signal_with_leak, signal_without_leak, sampling_rate)

    st.markdown("""
    ## Conclusion
    As you can see, when the signal length is not an integer multiple of the period, spectrum leakage occurs, resulting in a spread of the signal's energy across multiple frequency bins.
    """)


if __name__ == "__main__":
    main()

# %%
