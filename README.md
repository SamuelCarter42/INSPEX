# INSPEX

**In-Situ Electron Energy Spectral Analysis**

INSPEX is a user-friendly Python tool for downloading, analysing, and fitting in-situ electron energy spectra. It is intended as an in-situ equivalent to OSPEX for hard X-ray spectroscopy. The underlying methodology is described in Carter et al. (2025, in prep).

> **Note:** Loading STEREO STE data requires an installation of IDL, as INSPEX calls IDL functions internally. INSPEX functions normally without IDL for all other instruments and for data loaded via script.

---

## Requirements

INSPEX is developed using [Anaconda](https://www.anaconda.com/), and that distribution is recommended. The following additional packages are required:

| Package | Notes |
|---|---|
| `lmfit` | |
| `solo_epd_loader` | Gieseler & Palmroos (2025) |
| `pickle` | |
| `numdifftools` | |
| `spiceypy` | |
| `reproject` | |
| `sunpy_soar` | |
| `spacepy` | |
| `emcee` | |

---

## Usage

INSPEX can be used in two ways: through its interactive GUI (recommended for new users), or by calling it directly from a script with an externally generated spectrum.

### 1. GUI Workflow

#### Step 1 — Instrument & Data Selection

Call `instrument_choice()` at the console after running the INSPEX script. This opens a window where you can:

- Load data using INSPEX's built-in loaders for **SolO STEP**, **SolO EAS**, and **STEREO STE-D**
- Select a spectrum generation method: **instantaneous**, **peak flux**, or **fluence**
- Load a previously saved INSPEX spectrum (`.txt` format) to skip straight to fitting

> **Note:** Loading STEREO STE data requires an installation of IDL, as INSPEX calls IDL functions internally. INSPEX functions normally without IDL for all other instruments and for data loaded via script.

> **Note:** INSPEX does not remove negative flux values during loading, as these can be statistically meaningful. Before loading, verify data availability at the relevant archive. For Solar Orbiter in-situ instruments, check the [SolO data inventory](https://sites.google.com/view/solo-wg/information/data-in-situ-instruments).

#### Step 2 — Spectral Generation

Use sliders or entry fields to define the background range and integration time (or select instantaneous points). For the instantaneous method, multiple time points can be selected manually or at regular intervals, with each fitted in turn.

When using the joint cross-calibrated EAS/STEP product, spectra are cross-calibrated by computing a **Flux Alignment Factor (FAF)**: the ratio between the two instruments' average flux in their overlapping energy range, which is then applied to the EAS data.

#### Step 3 — Fitting

The fitting GUI opens alongside a live spectrum preview. You can:

- Build a model by selecting function components from a drop-down menu, or load a previously saved model
- Set energy range limits, initial parameter values, and upper/lower bounds
- Fix individual parameters by unchecking their *vary* box
- Preview the current initial guess before fitting
- Save the spectrum to avoid regenerating it in future sessions
- Run the minimiser and inspect the results and residuals in a separate window

Fitted parameter uncertainties are currently printed to the console and will be shown in the GUI in a future version. Fitted parameters can be saved to a `.txt` file, which also records the reduced chi-squared and BIC values for later reference.

### 2. Script Usage

To fit an externally generated spectrum, call:

```python
inspex.inspex(energies, fluxes, flux_uncertainties)
```

This opens the same fitting GUI described above.

---

## References

Gieseler, J., & Palmroos, C. (2025). *solo-epd-loader* (Version 0.4.4) [Computer software]. https://doi.org/10.5281/zenodo.15130823

Carter et al. (2025, in prep).
