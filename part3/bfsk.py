"""BFSK over screen alpha.

Bit 0 is a flicker at F0 hertz. Bit 1 is a flicker at F1 hertz.
Alpha is how opaque a black layer is on top of a still picture:
0 is the picture alone, and `delta` is the darkest the layer gets.
The phone sees that as a brightness wave.

Keep these constants in step with part3/bfsk.js.
"""

from __future__ import annotations

import math

import numpy as np

F0 = 5.0
F1 = 8.0
PREAMBLE = "10101101"
SYNC_THRESHOLD = 0.35


def encode_message(text: str) -> list[int]:
    """Preamble, one length byte, then the UTF-8 message, MSB first."""
    payload = text.encode("utf-8")
    if not payload:
        raise ValueError("Message is empty")
    if len(payload) > 255:
        raise ValueError("Message is longer than 255 bytes")
    bits = [int(bit) for bit in PREAMBLE]
    for shift in range(7, -1, -1):
        bits.append((len(payload) >> shift) & 1)
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def alpha_at(
    t: float,
    bits: list[int],
    bit_duration: float,
    delta: float,
    f0: float = F0,
    f1: float = F1,
) -> float:
    """Opacity at time t seconds from the start of the packet. Range 0..delta."""
    if t < 0 or not bits or t >= len(bits) * bit_duration:
        return 0.0
    index = min(int(t / bit_duration), len(bits) - 1)
    freq = f0 if bits[index] == 0 else f1
    local = t - index * bit_duration
    return (delta / 2.0) * (1.0 - math.cos(2.0 * math.pi * freq * local))


def bits_to_text(bits: list[int]) -> str:
    """Read a packet: skip the preamble, then one length byte, then the text."""
    preamble = [int(bit) for bit in PREAMBLE]
    if len(bits) < len(preamble) + 8:
        return ""
    cursor = len(preamble)
    length = 0
    for _ in range(8):
        length = (length << 1) | int(bits[cursor])
        cursor += 1
    data = bytearray()
    for _ in range(length):
        if cursor + 8 > len(bits):
            break
        byte = 0
        for _shift in range(8):
            byte = (byte << 1) | int(bits[cursor])
            cursor += 1
        data.append(byte)
    return data.decode("utf-8", errors="replace")


def tone_power(times: np.ndarray, values: np.ndarray, freq: float) -> float:
    """Energy of `values` at `freq`, timed by the real sample stamps."""
    if len(values) < 4:
        return 0.0
    centered = values - values.mean()
    phase = 2.0 * np.pi * freq * (times - times[0])
    sine = float(np.dot(centered, np.sin(phase)))
    cosine = float(np.dot(centered, np.cos(phase)))
    return sine * sine + cosine * cosine


def _highpass(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(values) < window:
        return values - values.mean()
    kernel = np.ones(window) / window
    return values - np.convolve(values, kernel, mode="same")


def find_start(
    times: np.ndarray,
    values: np.ndarray,
    template_bits: list[int],
    bit_duration: float,
    delta: float,
    f0: float = F0,
    f1: float = F1,
) -> tuple[float | None, float]:
    """Return (packet start time, match score in -1..1)."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) < 8:
        return None, 0.0
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return None, 0.0

    duration = len(template_bits) * bit_duration
    n_template = max(8, int(round(duration / dt)))
    rel = np.arange(n_template) * dt
    template = np.array(
        [-alpha_at(t, template_bits, bit_duration, delta, f0, f1) for t in rel]
    )
    smooth = max(3, int(round(0.8 / dt)))
    template = _highpass(template, smooth)
    template = template - template.mean()
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-8:
        return None, 0.0

    # A recording can start a fraction of a second after the packet. Search
    # that lead-in too, holding the first sample so the template can hang
    # off the front of the clip.
    lead = min(1.0, bit_duration * 2)
    grid = np.arange(float(times[0]) - lead, float(times[-1]), dt)
    if len(grid) < n_template + 2:
        return None, 0.0
    extended_times = np.concatenate(([grid[0]], times))
    extended_values = np.concatenate(([values[0]], values))
    signal = _highpass(np.interp(grid, extended_times, extended_values), smooth)
    corr = np.correlate(signal, template, mode="valid")
    count = len(template)
    sum_x = np.convolve(signal, np.ones(count), mode="valid")
    sum_x2 = np.convolve(signal * signal, np.ones(count), mode="valid")
    variance = np.maximum(sum_x2 / count - (sum_x / count) ** 2, 1e-12)
    scores = corr / (np.sqrt(variance * count) * template_norm)
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def classify_bits(
    times: np.ndarray,
    values: np.ndarray,
    start: float,
    n_bits: int,
    bit_duration: float,
    f0: float = F0,
    f1: float = F1,
) -> list[int]:
    got: list[int] = []
    for index in range(n_bits):
        left = start + index * bit_duration
        right = left + bit_duration
        selected = (times >= left) & (times < right)
        if int(selected.sum()) < 4:
            got.append(-1)
            continue
        power0 = tone_power(times[selected], values[selected], f0)
        power1 = tone_power(times[selected], values[selected], f1)
        got.append(0 if power0 >= power1 else 1)
    return got


def decode_samples(
    times: np.ndarray,
    values: np.ndarray,
    sent_bits: list[int],
    bit_duration: float,
    delta: float,
    f0: float = F0,
    f1: float = F1,
) -> dict:
    """Align `values` to the known packet and score every transmitted bit."""
    start, score = find_start(times, values, sent_bits, bit_duration, delta, f0, f1)
    duration = len(sent_bits) * bit_duration
    if start is None or score < SYNC_THRESHOLD:
        return {
            "ok": False,
            "message": "",
            "sent_bits": sent_bits,
            "got_bits": [],
            "n_bits": len(sent_bits),
            "n_correct": 0,
            "ber": 1.0,
            "data_rate_bps": 0.0,
            "sync_score": score,
            "duration_s": duration,
        }

    got = classify_bits(times, values, start, len(sent_bits), bit_duration, f0, f1)
    correct = sum(1 for sent, received in zip(sent_bits, got) if sent == received)
    return {
        "ok": True,
        "message": bits_to_text(got),
        "sent_bits": sent_bits,
        "got_bits": got,
        "n_bits": len(sent_bits),
        "n_correct": correct,
        "ber": 1.0 - correct / len(sent_bits),
        "data_rate_bps": correct / duration,
        "sync_score": score,
        "duration_s": duration,
    }
