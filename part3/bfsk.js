// BFSK over screen alpha. Keep the numbers in step with part3/bfsk.py.
(function (root) {
  const F0 = 5;
  const F1 = 8;
  const PREAMBLE = "10101101";
  const SYNC_THRESHOLD = 0.35;

  function encodeMessage(text) {
    const bytes = new TextEncoder().encode(text);
    if (bytes.length === 0) throw new Error("Message is empty");
    if (bytes.length > 255) throw new Error("Message is longer than 255 bytes");
    const bits = PREAMBLE.split("").map(Number);
    for (let shift = 7; shift >= 0; shift -= 1) bits.push((bytes.length >> shift) & 1);
    for (const byte of bytes) {
      for (let shift = 7; shift >= 0; shift -= 1) bits.push((byte >> shift) & 1);
    }
    return bits;
  }

  function alphaAt(t, bits, bitDuration, delta, f0 = F0, f1 = F1) {
    if (t < 0 || !bits.length || t >= bits.length * bitDuration) return 0;
    const index = Math.min(Math.floor(t / bitDuration), bits.length - 1);
    const freq = bits[index] === 0 ? f0 : f1;
    const local = t - index * bitDuration;
    return (delta / 2) * (1 - Math.cos(2 * Math.PI * freq * local));
  }

  function bitsToText(bits) {
    const preamble = PREAMBLE.split("").map(Number);
    if (bits.length < preamble.length + 8) return "";
    let cursor = preamble.length;
    let length = 0;
    for (let i = 0; i < 8; i += 1) length = (length << 1) | (bits[cursor++] === 1 ? 1 : 0);
    const data = [];
    for (let n = 0; n < length; n += 1) {
      if (cursor + 8 > bits.length) break;
      let byte = 0;
      for (let shift = 0; shift < 8; shift += 1) byte = (byte << 1) | (bits[cursor++] === 1 ? 1 : 0);
      data.push(byte);
    }
    return new TextDecoder().decode(new Uint8Array(data));
  }

  function median(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2) return sorted[mid];
    return 0.5 * (sorted[mid - 1] + sorted[mid]);
  }

  function mean(values) {
    let sum = 0;
    for (const value of values) sum += value;
    return sum / values.length;
  }

  function convolveSame(values, kernel) {
    const flipped = kernel.slice().reverse();
    const n = values.length;
    const m = flipped.length;
    const start = Math.floor((n + m - 1 - n) / 2);
    const out = new Array(n).fill(0);
    for (let i = 0; i < n; i += 1) {
      const idx = i + start;
      let sum = 0;
      for (let k = 0; k < m; k += 1) {
        const j = idx - k;
        if (j >= 0 && j < n) sum += values[j] * flipped[k];
      }
      out[i] = sum;
    }
    return out;
  }

  function highpass(values, window) {
    if (window < 3 || values.length < window) {
      const offset = mean(values);
      return values.map((value) => value - offset);
    }
    const kernel = new Array(window).fill(1 / window);
    const smooth = convolveSame(values, kernel);
    return values.map((value, index) => value - smooth[index]);
  }

  function tonePower(times, values, freq) {
    if (values.length < 4) return 0;
    const offset = mean(values);
    const t0 = times[0];
    let sine = 0;
    let cosine = 0;
    for (let i = 0; i < values.length; i += 1) {
      const phase = 2 * Math.PI * freq * (times[i] - t0);
      const centered = values[i] - offset;
      sine += centered * Math.sin(phase);
      cosine += centered * Math.cos(phase);
    }
    return sine * sine + cosine * cosine;
  }

  function findStart(times, values, templateBits, bitDuration, delta, f0 = F0, f1 = F1) {
    if (times.length < 8) return { start: null, score: 0 };
    const gaps = [];
    for (let i = 1; i < times.length; i += 1) gaps.push(times[i] - times[i - 1]);
    const dt = median(gaps);
    if (!(dt > 0)) return { start: null, score: 0 };

    const duration = templateBits.length * bitDuration;
    const nTemplate = Math.max(8, Math.round(duration / dt));
    const template = new Array(nTemplate);
    for (let i = 0; i < nTemplate; i += 1) {
      template[i] = -alphaAt(i * dt, templateBits, bitDuration, delta, f0, f1);
    }
    const smooth = Math.max(3, Math.round(0.8 / dt));
    const shaped = highpass(template, smooth);
    const templateMean = mean(shaped);
    for (let i = 0; i < shaped.length; i += 1) shaped[i] -= templateMean;
    let templateNorm = 0;
    for (const value of shaped) templateNorm += value * value;
    templateNorm = Math.sqrt(templateNorm);
    if (templateNorm < 1e-8) return { start: null, score: 0 };

    const lead = Math.min(1, bitDuration * 2);
    const grid = [];
    for (let t = times[0] - lead; t < times[times.length - 1]; t += dt) grid.push(t);
    if (grid.length < nTemplate + 2) return { start: null, score: 0 };
    const signal = new Array(grid.length);
    let cursor = 0;
    for (let i = 0; i < grid.length; i += 1) {
      const t = grid[i];
      while (cursor < times.length - 2 && times[cursor + 1] < t) cursor += 1;
      const t0 = times[cursor];
      const t1 = times[Math.min(cursor + 1, times.length - 1)];
      const span = t1 - t0 || 1;
      const mix = Math.min(1, Math.max(0, (t - t0) / span));
      const v0 = values[cursor];
      const v1 = values[Math.min(cursor + 1, values.length - 1)];
      signal[i] = v0 + (v1 - v0) * mix;
    }
    const filtered = highpass(signal, smooth);

    const count = shaped.length;
    const scores = [];
    let best = 0;
    let bestScore = -Infinity;
    for (let k = 0; k <= filtered.length - count; k += 1) {
      let corr = 0;
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < count; i += 1) {
        const value = filtered[k + i];
        corr += value * shaped[i];
        sum += value;
        sumSq += value * value;
      }
      const variance = Math.max(sumSq / count - (sum / count) ** 2, 1e-12);
      const score = corr / (Math.sqrt(variance * count) * templateNorm);
      scores.push(score);
      if (score > bestScore) {
        bestScore = score;
        best = k;
      }
    }
    return { start: grid[best], score: bestScore, scores };
  }

  function classifyBits(times, values, start, nBits, bitDuration, f0 = F0, f1 = F1) {
    const got = [];
    for (let index = 0; index < nBits; index += 1) {
      const left = start + index * bitDuration;
      const right = left + bitDuration;
      const sliceT = [];
      const sliceV = [];
      for (let i = 0; i < times.length; i += 1) {
        if (times[i] >= left && times[i] < right) {
          sliceT.push(times[i]);
          sliceV.push(values[i]);
        }
      }
      if (sliceV.length < 4) {
        got.push(-1);
        continue;
      }
      const power0 = tonePower(sliceT, sliceV, f0);
      const power1 = tonePower(sliceT, sliceV, f1);
      got.push(power0 >= power1 ? 0 : 1);
    }
    return got;
  }

  function decodeSamples(times, values, sentBits, bitDuration, delta, f0 = F0, f1 = F1) {
    const found = findStart(times, values, sentBits, bitDuration, delta, f0, f1);
    const duration = sentBits.length * bitDuration;
    if (found.start == null || found.score < SYNC_THRESHOLD) {
      return {
        ok: false,
        message: "",
        gotBits: [],
        nBits: sentBits.length,
        nCorrect: 0,
        ber: 1,
        dataRateBps: 0,
        syncScore: found.score || 0,
        durationS: duration,
      };
    }
    const got = classifyBits(times, values, found.start, sentBits.length, bitDuration, f0, f1);
    let correct = 0;
    for (let i = 0; i < sentBits.length; i += 1) if (sentBits[i] === got[i]) correct += 1;
    return {
      ok: true,
      message: bitsToText(got),
      gotBits: got,
      nBits: sentBits.length,
      nCorrect: correct,
      ber: 1 - correct / sentBits.length,
      dataRateBps: correct / duration,
      syncScore: found.score,
      durationS: duration,
    };
  }

  root.BFSK = {
    f0: F0,
    f1: F1,
    preamble: PREAMBLE,
    syncThreshold: SYNC_THRESHOLD,
    encodeMessage,
    alphaAt,
    bitsToText,
    findStart,
    classifyBits,
    decodeSamples,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
