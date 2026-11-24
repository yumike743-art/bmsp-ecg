"""Feature extraction on reference or detector sample positions."""

import hashlib

import numpy as np
from scipy import signal as dsp
from scipy.ndimage import median_filter

CLASSES = ("N", "S", "V", "F")
RR_NAMES = ("rr_previous_s", "rr_next_s", "rr_previous_over_local", "rr_next_over_local")
MORPH_NAMES = (
    "centre_mV", "qrs_min_mV", "qrs_max_mV", "qrs_peak_to_peak_mV", "qrs_rms_mV",
    "qrs_abs_area_mV_s", "qrs_half_peak_duration_s", "qrs_max_slope_mV_per_s",
    "window_rms_mV", "window_peak_to_peak_mV",
    *(f"mean_{start:+d}_to_{start + 30:+d}_ms_mV" for start in range(-180, 240, 30)),
)
FEATURE_NAMES = RR_NAMES + MORPH_NAMES


def symbol_classes(symbols, labels):
    lookup = {symbol: name for name, codes in labels.items() for symbol in codes}
    return np.array([lookup.get(symbol, "event") for symbol in symbols], dtype="U16")


def window_valid(centres, length, fs, window):
    before, after = (round(side * fs) for side in window)
    return (centres >= before) & (centres <= length - after)


def filter_sos(fs, spec):
    return dsp.butter(spec["prototype_order"], spec["band_hz"], btype="bandpass",
                      fs=fs, output="sos")


def bandpass(signal, fs, spec):
    return dsp.sosfiltfilt(filter_sos(fs, spec), signal)


def rr_features(centres, fs):
    intervals = np.diff(centres) / fs
    previous, following = np.r_[np.nan, intervals], np.r_[intervals, np.nan]
    if intervals.size:
        smooth = median_filter(intervals, size=11, mode="nearest")
        before_median, after_median = np.r_[np.nan, smooth], np.r_[smooth, np.nan]
    else:
        before_median = after_median = np.full(centres.size, np.nan)
        previous = following = before_median
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.column_stack([previous, following, previous / before_median, following / after_median])
    rr[~np.isfinite(rr)] = np.nan
    return rr


def feature_matrix(signal, centres, fs, window):
    """RR uses all given centres, including context outside the evaluation window."""
    valid = window_valid(centres, len(signal), fs, window)
    before, after = (round(side * fs) for side in window)
    offsets = np.arange(-before, after)
    beats = signal[centres[valid, None] + offsets]
    baseline = (offsets >= round(-0.20 * fs)) & (offsets < round(-0.12 * fs))
    beats = beats - np.median(beats[:, baseline], axis=1, keepdims=True)
    qrs_mask = (offsets >= round(-0.08 * fs)) & (offsets < round(0.10 * fs))
    qrs = beats[:, qrs_mask]
    absolute = np.abs(qrs)
    peak = absolute.max(axis=1)
    duration = np.sum((absolute >= 0.5 * peak[:, None]) & (peak[:, None] > 0), axis=1) / fs
    morphology = [
        beats[:, before], qrs.min(axis=1), qrs.max(axis=1), np.ptp(qrs, axis=1),
        np.sqrt(np.mean(qrs**2, axis=1)), absolute.sum(axis=1) / fs, duration,
        np.abs(np.diff(qrs, axis=1)).max(axis=1) * fs,
        np.sqrt(np.mean(beats**2, axis=1)), np.ptp(beats, axis=1),
    ]
    for ms in range(-180, 240, 30):
        mask = (offsets >= round(ms * fs / 1000)) & (offsets < round((ms + 30) * fs / 1000))
        morphology.append(beats[:, mask].mean(axis=1))
    return np.column_stack([rr_features(centres, fs)[valid], *morphology]), valid


def unit_noise(record, kind, seed, drift_hz):
    digest = hashlib.sha256(f"{seed}/{record.split.record_id}/{kind}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    if kind == "white":
        noise = rng.standard_normal(record.signal.size)
    elif kind == "drift":
        phase = rng.uniform(0, 2 * np.pi)
        seconds = np.arange(record.signal.size) / record.fs
        noise = np.sin(2 * np.pi * drift_hz * seconds + phase)
    else:
        raise ValueError(f"Unknown noise type: {kind}")
    return noise - noise.mean()


def inject(signal, noise, snr_db):
    signal_power = np.mean((signal - signal.mean())**2)
    noise_power = np.mean(noise**2)
    if min(signal_power, noise_power) <= 0:
        raise ValueError("SNR requires positive signal and noise power")
    delta = noise * np.sqrt(signal_power / (noise_power * 10**(snr_db / 10)))
    noisy = signal + delta
    actual_db = 10 * np.log10(signal_power / np.mean((noisy - signal)**2))
    return noisy, float(actual_db)
