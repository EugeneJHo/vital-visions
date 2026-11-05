#!/usr/bin/env python3
"""Part 1: heart rate from a fingertip video recorded with the phone flash on.

Steps, matching the assignment:
  1. Average each frame after converting it to grayscale (MATLAB rgb2gray weights).
  2. Subtract the DC value (the mean of that trace).
  3. High-pass filter to remove the slow drift from finger pressure.
  4. FFT the filtered trace and take the strongest bin between 60 and 100 bpm.
  5. Find one peak per heartbeat. The minimum gap between peaks is set from
     the heart rate in step 4.

Run from this folder:
  .venv/bin/python heart_rate.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep matplotlib's cache inside the project so it does not need a home-dir write.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent
VIDEO_PATH = ROOT / "IMG_2867.MOV"
OUTPUT_DIR = ROOT / "output"

# Cardiac band the assignment asks us to search: a resting heart rate.
HR_MIN_BPM = 60.0
HR_MAX_BPM = 100.0

# High-pass cutoff sits below 60 bpm (1 Hz), so the pulse stays and the
# slow baseline (breathing, pressing harder) is removed.
HIGHPASS_HZ = 0.7
HIGHPASS_ORDER = 2

# A full beat-period gap drops beats that arrive a little early. 80% of the
# estimated period still rejects the small secondary bump within a beat.
PEAK_DISTANCE_FRACTION = 0.8


def read_mean_grayscale(path: Path) -> tuple[np.ndarray, float]:
    """Mean rgb2gray value of every frame, plus the video frame rate in Hz."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    means: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV is BGR. These are the MATLAB rgb2gray coefficients.
        blue = frame[:, :, 0].astype(np.float64)
        green = frame[:, :, 1].astype(np.float64)
        red = frame[:, :, 2].astype(np.float64)
        gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
        means.append(float(gray.mean()))
    cap.release()

    if len(means) < 10:
        raise RuntimeError(f"Video has too few frames: {len(means)}")
    return np.asarray(means), fps


def highpass(signal: np.ndarray, fps: float, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth high-pass. filtfilt avoids shifting the peaks."""
    nyquist = 0.5 * fps
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"Cutoff {cutoff_hz} Hz is outside (0, {nyquist}) for fps={fps}")
    b, a = butter(order, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, signal)


def dominant_heart_rate(signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Single-sided FFT. Return freqs (Hz), magnitude, and the peak freq in the 60–100 bpm band."""
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    low = HR_MIN_BPM / 60.0
    high = HR_MAX_BPM / 60.0
    band = (freqs >= low) & (freqs <= high)
    if not np.any(band):
        raise RuntimeError("FFT band for 60–100 bpm is empty. The clip may be too short.")

    peak_freq = float(freqs[band][np.argmax(spectrum[band])])
    return freqs, spectrum, peak_freq


def main() -> None:
    raw, fps = read_mean_grayscale(VIDEO_PATH)
    # Step 2: remove DC. Step 3: high-pass.
    dc_removed = raw - raw.mean()
    filtered = highpass(dc_removed, fps, HIGHPASS_HZ, HIGHPASS_ORDER)

    freqs, spectrum, peak_hz = dominant_heart_rate(filtered, fps)
    fft_bpm = peak_hz * 60.0

    # Samples between peaks at the estimated rate: fs / f = fs * 60 / bpm.
    # The prompt's "normally 30 Hz" figure is printed below; the video's own
    # frame rate is what the peak finder has to use.
    period_samples = fps / peak_hz
    period_samples_at_30hz = 30.0 / peak_hz
    min_peak_distance = max(1, int(round(PEAK_DISTANCE_FRACTION * period_samples)))

    peak_indices, _ = find_peaks(filtered, distance=min_peak_distance)
    if len(peak_indices) < 2:
        raise RuntimeError("Need at least two peaks to compute a heart rate.")

    intervals_s = np.diff(peak_indices) / fps
    heart_rate_bpm = 60.0 / float(intervals_s.mean())
    time_s = np.arange(len(filtered)) / fps

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_fft(freqs, spectrum, peak_hz, OUTPUT_DIR / "fft_heart_rate.png")
    plot_peaks(
        time_s,
        filtered,
        peak_indices,
        heart_rate_bpm,
        OUTPUT_DIR / "peaks_heart_rate.png",
    )

    print(f"Video: {VIDEO_PATH.name}")
    print(f"Frames: {len(raw)}    duration: {len(raw) / fps:.2f} s    frame rate: {fps:.3f} Hz")
    print(f"FFT peak in 60–100 bpm: {peak_hz:.3f} Hz  ({fft_bpm:.1f} bpm)")
    print(
        "MinPeakDistance: one beat is "
        f"{period_samples:.1f} samples at {fps:.1f} Hz "
        f"({period_samples_at_30hz:.1f} samples at the usual 30 Hz). "
        f"Using {min_peak_distance} samples "
        f"({PEAK_DISTANCE_FRACTION:.0%} of that period)."
    )
    print(
        f"Peaks found: {len(peak_indices)}    "
        f"mean interval: {intervals_s.mean() * 1000:.0f} ms    "
        f"heart rate from peaks: {heart_rate_bpm:.1f} bpm"
    )
    print(f"Wrote {OUTPUT_DIR / 'fft_heart_rate.png'}")
    print(f"Wrote {OUTPUT_DIR / 'peaks_heart_rate.png'}")


def plot_fft(freqs: np.ndarray, spectrum: np.ndarray, peak_hz: float, path: Path) -> None:
    low = HR_MIN_BPM / 60.0
    high = HR_MAX_BPM / 60.0
    show = freqs <= 4.0

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axvspan(low, high, color="#f4c7c3", alpha=0.85, label="60–100 bpm", zorder=0)
    ax.plot(freqs[show], spectrum[show], color="#7a1f2b", lw=1.6, label="FFT magnitude")
    peak_mag = float(spectrum[np.argmin(np.abs(freqs - peak_hz))])
    ax.plot(peak_hz, peak_mag, "o", color="#1f4b99", ms=7, zorder=3)
    ax.annotate(
        f"{peak_hz:.2f} Hz\n({peak_hz * 60:.1f} bpm)",
        xy=(peak_hz, peak_mag),
        xytext=(peak_hz + 0.35, peak_mag * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#1f4b99"},
        color="#1f4b99",
        fontsize=11,
    )
    ax.set_xlim(0, 4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"FFT of the filtered heart-rate signal — {peak_hz:.2f} Hz")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_peaks(
    time_s: np.ndarray,
    signal: np.ndarray,
    peak_indices: np.ndarray,
    heart_rate_bpm: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(time_s, signal, color="#7a1f2b", lw=1.15, label="Filtered signal")
    ax.plot(
        time_s[peak_indices],
        signal[peak_indices],
        "o",
        color="#1f4b99",
        ms=5.5,
        label="Detected beats",
    )
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Grayscale (DC removed, high-pass)")
    ax.set_title(f"Heart-rate signal with detected beats — {heart_rate_bpm:.1f} bpm")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
