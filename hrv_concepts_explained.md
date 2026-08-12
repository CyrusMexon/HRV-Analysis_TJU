# HRV Frequency Domain Concepts Explained

## Understanding the Fundamentals

### 1. What is "Sampling" and "Evenly Sampled RR"?

#### Raw RR Intervals (Unevenly Sampled)
When you measure heart beats, you get RR intervals like:
```
Time:     0s      0.8s    1.6s    2.4s    3.2s
RR:       800ms   820ms   790ms   810ms   805ms
```

These are **unevenly spaced in time** because:
- Each beat occurs at a different time
- The time between measurements varies (it's the RR interval itself!)
- This is called "event-based" or "unevenly sampled" data

#### Problem: FFT Requires Evenly Sampled Data
FFT (Fast Fourier Transform) requires data points at **regular time intervals**:
```
Time:     0s    0.25s   0.5s   0.75s   1.0s   1.25s  ...
Value:    ?     ?       ?      ?       ?      ?
```

#### Solution: Interpolation & Resampling
We convert RR intervals to an **evenly sampled time series**:

**Step 1: Create time points from RR intervals**
```python
# RR intervals in seconds
rr_s = [0.800, 0.820, 0.790, 0.810, 0.805]

# Calculate when each beat occurred
beat_times = [0, 0.800, 1.620, 2.410, 3.220]
```

**Step 2: Interpolate to create a continuous signal**
```python
# Use cubic spline to estimate RR value at ANY time
# This creates a smooth curve through the beat points
interpolated_function = cubic_spline(beat_times, rr_s)
```

**Step 3: Resample at regular intervals**
```python
sampling_rate = 4 Hz  # 4 samples per second
dt = 1/4 = 0.25 seconds

# Now we sample at regular intervals
new_times = [0, 0.25, 0.50, 0.75, 1.00, 1.25, ...]
new_values = [interpolated_function(t) for t in new_times]

# Result: Evenly sampled signal!
Time:   0.00    0.25    0.50    0.75    1.00    1.25
Value:  0.800   0.805   0.812   0.808   0.803   0.798  (seconds)
```

#### Why 4 Hz Sampling Rate?
- **Nyquist theorem**: To detect frequency `f`, you need sampling rate ≥ 2f
- HRV highest frequency of interest: ~0.4 Hz (HF band upper limit)
- Minimum sampling: 2 × 0.4 = 0.8 Hz
- Standard uses 4 Hz for good margin (can detect up to 2 Hz)

**Visual Example:**
```
Unevenly sampled (raw RR):
    |        |       |        |       |
    800ms    820ms   790ms    810ms   805ms

Evenly sampled at 4 Hz (after interpolation):
    |    |    |    |    |    |    |    |
   0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 seconds
```

---

### 2. What is "Window Width" and Why 256 seconds?

#### The Problem: Signal is Too Long
Your RR signal might be 5 minutes (300 seconds) long.
- Computing FFT on the entire signal gives **very fine frequency resolution**
- But can be **noisy** and sensitive to non-stationarity
- Heart rate isn't perfectly stationary over long periods

#### The Solution: Welch's Method - Divide into Windows

**Welch's Method** divides the signal into smaller segments (windows):

```
Original signal (300 seconds):
|--------------------------------------------------|

Divide into windows of 256 seconds:
|------------------------------|
                    |------------------------------|
                                        |------------------------------|

With 50% overlap:
|------------------------------|
              |------------------------------|
                            |------------------------------|
```

**Why use windows?**
1. **Reduce variance** (noise) by averaging multiple estimates
2. **Handle non-stationarity** better (heart rate changes over time)
3. **Trade-off** between frequency resolution and variance

#### Window Width = 256 seconds

**Window width affects frequency resolution:**
```
Frequency resolution = 1 / window_width

256s window → resolution = 1/256 = 0.0039 Hz
120s window → resolution = 1/120 = 0.0083 Hz
```

**Smaller resolution = Better precision** in identifying frequencies

**Why 256?**
- Power of 2 (makes FFT computationally efficient)
- Long enough for good frequency resolution in VLF/LF bands
- Common standard in HRV analysis
- For VLF band (0.003-0.04 Hz), need window ≥ 1/0.003 ≈ 333s ideally
- 256s is a practical compromise

#### Example:
```
If your signal is 61.82 seconds (your data):
- Cannot use 256s window (signal too short!)
- Must use entire signal (~62s) as one window
- Frequency resolution = 1/62 = 0.016 Hz (coarser)
```

---

### 3. What is "Window Overlap" and Why 50%?

#### Without Overlap
Divide signal into non-overlapping segments:
```
Segment 1:  [0 - 256s]
Segment 2:  [256 - 512s]
Segment 3:  [512 - 768s]
```
- Uses all data exactly once
- Might miss information at segment boundaries

#### With 50% Overlap
Each segment overlaps with the next by 50%:
```
Segment 1:  [0   - 256s]
Segment 2:  [128 - 384s]  ← Starts at 50% of previous
Segment 3:  [256 - 512s]
Segment 4:  [384 - 640s]
```

**Benefits:**
1. **More segments to average** → Lower variance (smoother PSD)
2. **Data at boundaries** gets used in multiple segments
3. **Better statistical properties**

**Cost:**
- Segments are not independent (correlated)
- Slightly biased variance reduction
- But empirically works well in practice

**Why 50%?**
- Good balance between variance reduction and computational cost
- Common standard (also 0%, 66.7%, 75% are used)
- More overlap = more segments = smoother result (but diminishing returns)

---

### 4. What is a "Window Function" and Why Hann?

#### The Problem: Spectral Leakage

When you cut a signal into segments, you create **discontinuities** at edges:

```
Original smooth signal:
    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿

Cut into segment:
    |∿∿∿∿∿∿∿∿∿∿∿∿∿∿|
    ^              ^
    Jump!          Jump!
```

These jumps create **false high frequencies** in the FFT (spectral leakage).

#### The Solution: Window Functions

Multiply signal by a window that **smoothly goes to zero at edges**:

**Hann Window:**
```
Weight
1.0 |        ╱‾‾‾‾‾‾‾‾╲
    |      ╱            ╲
0.5 |    ╱                ╲
    |  ╱                    ╲
0.0 |╱________________________╲
    0                      256 seconds

Formula: w(n) = 0.5 - 0.5*cos(2πn/N)
```

**After applying window:**
```
Windowed signal:
    |  ∿∿∿∿∿∿∿∿∿∿  |
    ╱                ╲    ← Smoothly goes to zero
```

**Common Window Types:**
- **Rectangular**: No window (just cut) - maximum leakage
- **Hann**: Good general-purpose, smooth
- **Hamming**: Similar to Hann, slightly different shape
- **Blackman**: Very smooth, less leakage, wider main lobe

**Why Hann?**
- Good compromise between:
  - Main lobe width (frequency resolution)
  - Side lobe suppression (leakage reduction)
- Standard in HRV analysis
- Well-studied properties

---

### 5. How Window Width Affects Values

#### Effect on Frequency Resolution
```
Window Width     Freq Resolution     Can Distinguish
256 seconds      0.0039 Hz          0.02 Hz vs 0.024 Hz ✓
120 seconds      0.0083 Hz          0.02 Hz vs 0.024 Hz ?
60 seconds       0.0167 Hz          0.02 Hz vs 0.024 Hz ✗
```

**For VLF band (0.003 - 0.04 Hz):**
- Small window: Might not clearly separate VLF from DC or LF
- Large window: Better separation of close frequencies

#### Effect on Power Values

**Longer Window:**
- **Finer frequency resolution** (more frequency bins)
- **Higher variance** (noisier estimate)
- Uses more data per estimate
- Fewer segments to average (if signal length is fixed)

**Shorter Window:**
- **Coarser frequency resolution** (fewer bins)
- **Lower variance** (smoother estimate through averaging)
- More segments to average
- Better for non-stationary signals

#### Why This Affects Your Values

Let's compare for your 61.82-second signal:

**Scenario 1: Entire signal as one window (61.82s)**
```
Number of windows: 1
Frequency resolution: 0.016 Hz
Averaging: None (single estimate)
Result: Higher variance, but uses all data
```

**Scenario 2: Try to use 256s window (impossible - signal too short!)**
```
Number of windows: Can't fit even one!
Solution: Use entire signal (falls back to Scenario 1)
```

**Scenario 3: If signal was 600 seconds, use 256s with 50% overlap**
```
Window 1: [0-256]
Window 2: [128-384]
Window 3: [256-512]
Window 4: [384-600] (partial)

Number of windows: ~3-4
Frequency resolution: 0.0039 Hz
Averaging: 3-4 estimates
Result: Smoother PSD, better variance
```

---

### 6. Putting It All Together - Your Data

**Your data file:** 77 RR intervals, 61.82 seconds

**Step 1: Interpolation & Resampling**
```
Input: 77 unevenly-spaced RR intervals
↓ Cubic spline interpolation
↓ Resample at 4 Hz
Output: 247 evenly-spaced samples (61.82 × 4 ≈ 247)
```

**Step 2: Welch's Method with Kubios Settings**
```
Window width: 256 seconds
Your signal: 61.82 seconds
→ Signal too short! Use entire signal as one window

Window overlap: 50%
→ Doesn't matter (only one window)

Window function: Hann
→ Applied to the 247 samples
```

**Step 3: Compute FFT**
```
Windowed signal: 247 samples
↓ FFT
Output: 124 frequency bins (0 to 2 Hz)
Frequency resolution: 4/247 = 0.016 Hz
```

**Step 4: Extract Band Powers**
```
VLF (0.003-0.04 Hz):
  Bins in range: 0.016, 0.032 Hz (only 2 bins!)
  Integrate PSD over these frequencies

LF (0.04-0.15 Hz):
  Bins in range: 0.048, 0.064, 0.081, 0.097, 0.113, 0.129, 0.145
  More bins = better estimate
```

**Why VLF might be inaccurate for your data:**
- Only 2 frequency bins in VLF range!
- Frequency resolution (0.016 Hz) is almost as wide as VLF band (0.037 Hz)
- Signal too short for reliable VLF estimation
- This is why standards recommend ≥5 minutes for VLF analysis

---

### 7. Summary

| Concept | Meaning | Why It Matters |
|---------|---------|----------------|
| **Sampling Rate** | How often we measure (4 Hz = 4 times/second) | Must be ≥2× highest frequency of interest |
| **Evenly Sampled** | Data points at regular intervals | Required for FFT |
| **Interpolation** | Estimating values between measurements | Converts unevenly-spaced RR to evenly-spaced signal |
| **Window Width** | Length of each segment (256s) | Affects frequency resolution (1/width) |
| **Window Overlap** | How much segments overlap (50%) | More overlap = more averages = smoother |
| **Window Function** | Hann, Hamming, etc. | Reduces spectral leakage at segment edges |

---

### 8. Practical Guidelines

**For Reliable VLF Analysis:**
- Need at least 5 minutes (300s) of data
- Use window ≥ 256s (ideally 300s+)
- Frequency resolution should be < 0.01 Hz

**For Your 62-second Signal:**
- VLF estimates will be unreliable (too short)
- LF and HF should be okay
- Consider collecting longer recordings

**Choosing Settings:**
- **Short recordings (<2 min)**: Use entire signal, no averaging
- **Medium (2-5 min)**: 120s windows, 50% overlap
- **Long (≥5 min)**: 256-300s windows, 50% overlap

---

Would you like me to visualize any of these concepts or test your data with the exact Kubios settings (256s window, 50% overlap)?
