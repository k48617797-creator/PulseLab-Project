"""
PulseLab College Experiment: Time-Domain & FFT Spectral Analysis
Waveform: COMPOSITE | fs = 256 Hz | Window: hann
"""
import numpy as np
import matplotlib.pyplot as plt

# 1. Synthesize Discrete-Time Signal
fs = 256          # Sampling frequency (Hz)
duration = 1.00    # Signal duration (s)
N = int(fs * duration)
t = np.arange(N) / fs
np.random.seed(42)
clean = 0 + 1.4 * np.sin(2 * np.pi * 12 * t) + 0.65 * np.sin(2 * np.pi * 42 * t)
x = clean + 0.12 * np.random.randn(N)

# 2. Apply Window & Compute Single-Sided FFT
x_proc = x - np.mean(x) if True else x
w = np.hanning(N)
coherent_gain = np.mean(w)

n_fft = max(512, 1 << int(np.ceil(np.log2(N))))
X = np.fft.rfft(x_proc * w, n=n_fft)
freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
mag = (2.0 / (N * coherent_gain)) * np.abs(X)
mag[0] *= 0.5
psd_db = 10.0 * np.log10(np.maximum(1e-12, (np.abs(X)**2) / (fs * np.sum(w**2)) * 2.0))

dom_idx = np.argmax(mag[1:]) + 1
print(f"Samples N          : {N}")
print(f"RMS Amplitude      : {np.sqrt(np.mean(x**2)):.4f} V")
print(f"Signal Energy      : {np.sum(x**2):.3f} V^2")
print(f"Dominant Frequency : {freqs[dom_idx]:.2f} Hz (Mag: {mag[dom_idx]:.3f} V)")
print(f"Nyquist Frequency  : {fs / 2:.2f} Hz")

# 3. Publication-Grade Matplotlib Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.2), dpi=130)
fig.patch.set_facecolor('#ffffff')

ax1.plot(t, x, color='#2563eb', linewidth=1.3, label='Observed x[n]')
ax1.plot(t, clean, color='#94a3b8', linewidth=0.9, linestyle='--', alpha=0.8, label='Clean reference')
ax1.set_title('Time-Domain Signal x(t)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (V)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, fontsize=8.5)

ax2.plot(freqs, mag, color='#0891b2', linewidth=1.4, label='|X(f)| Amplitude Spectrum')
ax2.fill_between(freqs, mag, color='#0891b2', alpha=0.14)
ax2.axvline(freqs[dom_idx], color='#d97706', linestyle='--', linewidth=1.1,
            label=f'Peak: {freqs[dom_idx]:.2f} Hz')
ax2.set_title('Single-Sided FFT Magnitude Spectrum (hann window)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude (V)')
ax2.set_xlim(0, fs / 2)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True, fontsize=8.5)

plt.tight_layout(pad=1.6)
plt.show()
