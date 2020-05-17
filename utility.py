import numpy as np
from scipy.stats import skew
from scipy.signal import welch


eps = 0.00000001

# Computes zero crossing rate of frame
def zero_crossing_rate(frame):
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)


# Computes the spectral entropy (time and frequency)
def spectral_entropy(signal, n_short_blocks=10):
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


# Computes spectral centroid of frame (given abs(FFT))
def spectral_centroid(fft_magnitude, sampling_rate=40):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    centroid = (NUM / DEN)

    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)

    return centroid


# Computes spectral centroid of frame (given abs(FFT))
def spectral_spread(fft_magnitude, sampling_rate=40):
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Spread:
    spread = np.sqrt(np.sum(((ind - (NUM / DEN)) ** 2) * Xt) / DEN)

    # Normalize:
    spread = spread / (sampling_rate / 2.0)

    return spread


# Computes the spectral flux feature of the current frame
def spectral_flux(fft_magnitude, previous_fft_magnitude):
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum((fft_magnitude / fft_sum - previous_fft_magnitude / previous_fft_sum) ** 2)

    return sp_flux


# Computes spectral roll-off
def spectral_rolloff(signal, c=0.90):
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    # Find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    sp_rolloff = 0.0
    if len(a) > 0: sp_rolloff = np.float64(a[0]) / (float(fft_length))

    return sp_rolloff


def skewness(signal):
    return skew(signal)


def avg_power(signal):
    _, power = welch(signal, 40)
    return np.mean(power)


def rms(signal):
    return np.sqrt(np.mean(signal ** 2))
