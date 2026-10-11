# MNN Audio Module

The MNN audio module provides efficient audio processing operations optimized for edge devices, including audio I/O, feature extraction, and common transformations used in speech recognition and audio analysis.

## Installation

The audio module is included in the MNN Python package:

```bash
pip install MNN
```

## Quick Start

```python
import MNN.audio as audio

# Load audio file (filename, sr, frame_offset, num_frames)
waveform, sample_rate = audio.load("audio.wav", 0, 0, -1)

# Extract mel-spectrogram features (waveform, n_mels, n_fft, sample_rate)
# Note: mel_spectrogram binding only accepts up to 4 int parameters
mel_spec = audio.mel_spectrogram(waveform, 80, 400, 16000)
```

## API Reference

### Audio I/O

#### load()

Load audio from file with optional resampling.

```python
waveform, sample_rate = audio.load(filename, sr, frame_offset, num_frames)
```

**Parameters (all positional):**
- `filename` (str): Path to audio file (supports wav, mp3, flac, etc.)
- `sr` (int): Resample to this rate (0 keeps original)
- `frame_offset` (int): Start loading from this frame
- `num_frames` (int): Load this many frames (-1 for entire file)

**Returns:** `(waveform, sample_rate)` tuple

**Example:**
```python
# Load entire file at original rate
waveform, sr = audio.load("input.wav", 0, 0, -1)

# Resample to 16kHz
waveform, sr = audio.load("input.wav", 16000, 0, -1)
```
#### save()

Save audio waveform to file.

```python
audio.save(filename, waveform, sample_rate)
```

**Parameters:**
- `filename` (str): Output file path
- `waveform` (Var): Audio data as MNN Var
- `sample_rate` (int): Sample rate in Hz

---

### Feature Extraction

#### spectrogram()

Compute power or magnitude spectrogram.

```python
spec = audio.spectrogram(
    waveform,
    n_fft,
    hop_length,      # 0 = n_fft // 4
    win_length,      # 0 = n_fft
    window_type,     # int: 0=hann, 1=hamming, etc.
    pad_left,
    pad_right,
    center,          # int: 0 or 1
    normalized,      # int: 0 or 1
    pad_mode,        # int: see audio.h
    power            # 2.0 = power, 1.0 = magnitude
)
```

**Parameters (all positional):**
- `waveform`: Input audio tensor
- `n_fft` (int): FFT size
- `hop_length` (int): Hop size (0 = n_fft/4)
- `win_length` (int): Window size (0 = n_fft)
- `window_type` (int): Window function (0=hann, 1=hamming, etc.)
- `pad_left`, `pad_right` (int): Padding
- `center` (int): Center padding flag (0 or 1)
- `normalized` (int): Normalize flag (0 or 1)
- `pad_mode` (int): Padding mode
- `power` (float): Exponent (2.0 for power, 1.0 for magnitude)

**Example:**
```python
spec = audio.spectrogram(waveform, 400, 160, 400, 0, 0, 0, 1, 0, 0, 2.0)
```

#### mel_spectrogram()

Compute mel-scale spectrogram.

```python
# WARNING: The binding only accepts 4 int parameters due to format string bug
mel_spec = audio.mel_spectrogram(waveform, n_mels, n_fft, sample_rate)
```

**Parameters (all positional):**
- `waveform`: Input audio tensor
- `n_mels` (int): Number of mel filterbanks
- `n_fft` (int): FFT size
- `sample_rate` (int): Audio sample rate

**Note:** The Python binding's format string limits this to 4 integer parameters. Additional parameters (htk, norm, f_min, f_max, hop_length, etc.) are NOT accessible from Python.

**Example:**
```python
mel_spec = audio.mel_spectrogram(waveform, 80, 400, 16000)
```

#### fbank()

Extract Fbank features (Kaldi-style, commonly used in ASR).

```python
fbank_features = audio.fbank(
    waveform,
    sample_rate,
    n_mels,
    n_fft,
    hop_length,
    dither,
    preemphasis
)
```

**Parameters (all positional):**
- `waveform`: Input audio tensor
- `sample_rate` (int): Audio sample rate
- `n_mels` (int): Number of mel filterbanks
- `n_fft` (int): FFT size
- `hop_length` (int): Hop size
- `dither` (float): Dithering coefficient
- `preemphasis` (float): Pre-emphasis coefficient (typical: 0.97)

**Example:**
```python
fbank = audio.fbank(waveform, 16000, 80, 400, 160, 0.0, 0.97)
```

#### whisper_fbank()

Extract Whisper-compatible Fbank features.

```python
whisper_features = audio.whisper_fbank(
    waveform,
    sample_rate,
    n_mels,
    n_fft,
    hop_length,
    chunk_len
)
```

**Parameters (all positional):**
- `waveform`: Input audio tensor
- `sample_rate` (int): Audio sample rate (Whisper uses 16000)
- `n_mels` (int): Number of mel bins (Whisper uses 128)
- `n_fft` (int): FFT size
- `hop_length` (int): Hop size
- `chunk_len` (int): Chunk length in seconds

**Example:**
```python
whisper_fbank = audio.whisper_fbank(waveform, 16000, 128, 400, 160, 30)
```

---

### Window Functions

#### hamming_window()

Generate Hamming window.

```python
window = audio.hamming_window(window_size, periodic, alpha, beta)
```

**Parameters (all positional):**
- `window_size` (int): Window length
- `periodic` (int): Periodic flag (0 or 1)
- `alpha` (float): Alpha coefficient (default: 0.54)
- `beta` (float): Beta coefficient (default: 0.46)

**Example:**
```python
hamming = audio.hamming_window(400, 1, 0.54, 0.46)
```

#### hann_window()

Generate Hann window.

```python
window = audio.hann_window(window_size, periodic)
```

**Parameters (all positional):**
- `window_size` (int): Window length
- `periodic` (int): Periodic flag (0 or 1)

**Example:**
```python
hann = audio.hann_window(400, 1)
```

