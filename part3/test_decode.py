"""Synthetic camera signal. A clean run must come back with no bit errors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bfsk import F0, F1, decode_samples, encode_message


def synthesize(bits, bit_duration, delta, fps, lead_s, tail_s, noise, ramp):
    duration = lead_s + len(bits) * bit_duration + tail_s
    times = np.arange(0.0, duration, 1.0 / fps)
    from bfsk import alpha_at

    alpha = np.array(
        [
            alpha_at(t - lead_s, bits, bit_duration, delta, F0, F1) if t >= lead_s else 0.0
            for t in times
        ]
    )
    # Black overlay: higher alpha, lower brightness. Plus exposure drift and noise.
    rng = np.random.default_rng(0)
    brightness = 170.0 - 90.0 * alpha + ramp * times + rng.normal(0.0, noise, size=len(times))
    return times, brightness


def check(label, **kwargs):
    bits = encode_message("HELLO")
    times, values = synthesize(bits, **kwargs)
    result = decode_samples(times, values, bits, kwargs["bit_duration"], kwargs["delta"])
    print(
        f"{label}: message={result['message']!r} ber={result['ber']:.3f} "
        f"rate={result['data_rate_bps']:.2f} sync={result['sync_score']:.2f}"
    )
    if result["message"] != "HELLO" or result["ber"] != 0.0:
        raise SystemExit(f"decode failed for {label}")


def main():
    check("30fps d0.5", bit_duration=0.4, delta=0.5, fps=30, lead_s=1.5, tail_s=0.8, noise=1.5, ramp=0.4)
    check("24fps d0.1", bit_duration=0.4, delta=0.1, fps=24, lead_s=2.0, tail_s=1.0, noise=0.8, ramp=0.15)
    check("30fps fast", bit_duration=0.25, delta=0.5, fps=30, lead_s=1.0, tail_s=0.5, noise=1.0, ramp=0.2)
    print("ok")


if __name__ == "__main__":
    main()
