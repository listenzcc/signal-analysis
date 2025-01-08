# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_spectrum_leak(signal, sampling_rate, ax, length, hue_norm, alpha=0.5):
    signal = signal.copy()[:length]
    # Compute the FFT of the signal
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    fft_result = fft_result[np.abs(fft_freq) < 100]
    fft_freq = fft_freq[np.abs(fft_freq) < 100]

    # Plot using seaborn
    sns.lineplot(x=fft_freq, y=np.abs(fft_result),
                 ax=ax, hue=length, hue_norm=hue_norm, alpha=alpha, palette='RdBu')
    plt.title('Spectrum Leak', fontname='Comic Sans MS',
              fontsize=20)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    return plt.gcf()


freq = 50  # Hz
sampling_rate = 1000  # 1000 Hz
length = 500

t = np.linspace(0, 1, sampling_rate, endpoint=False)

# --
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
ax = plt.gca()
signal = np.sin(2 * np.pi * freq * t)
for length in range(800, 820):
    plot_spectrum_leak(signal, sampling_rate, ax,
                       length, (800, 820), alpha=0.5)
plt.show()

# --
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
ax = plt.gca()
signal = np.sin(2 * np.pi * freq * t)
for length in range(200, 220):
    plot_spectrum_leak(signal, sampling_rate, ax,
                       length, (200, 220), alpha=0.5)
plt.show()

# %% --
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
ax = plt.gca()
signal = np.sin(2 * np.pi * freq * t)
for length in range(200, 220, 2):
    plot_spectrum_leak(signal, sampling_rate, ax,
                       length, (200, 820), alpha=0.5)
for length in range(800, 820, 2):
    plot_spectrum_leak(signal, sampling_rate, ax,
                       length, (200, 820), alpha=0.5)
ax.set_ylim([0, 50])
plt.show()

# %%
