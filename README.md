# XTG @ MID: scan-by-scan analysis GUI (scaffold)

A PyQt5 + pyqtgraph GUI for the XTG experiment at European XFEL MID. It runs on **processed data, offline, scan by scan**; MID staff handle the live monitoring.
It starts on real data (`extra_data`) and loads nothing until you press **Load** or pass `--scans`; `--backend simulated` gives fake data for testing. Source names not known yet are **placeholders**. All the names live in one config file, so the real ones can drop in later without code changes.

## Proposals

The **proposal** selector (Data card, or `--preset` / `--proposal` on the command line) switches between:

- **11456 (this beamtime)**: two Jungfraus, the pump / probe diamond diodes, **one** upstream spectrometer before the SDL (page in a 2 × 2 grid), CRL pointing. Sources come from `Sources_11456.yaml` in the folder you start the GUI from; without it, the template `xtg_mid_gui/presets/Sources_11456.yaml` (placeholders) is used. If that file lists more spectrometers, only the first is used.
- **7016 (previous beamtime)**: sources from `Sources_7016.yaml` (shipped in `xtg_mid_gui/presets/`; a copy in the start folder takes precedence), as in Den's `rocking_MID.ipynb`:
  - **AGIPD1M, all 16 tiles**, laid out as in Den's `show_detector_multi` figure (tiles across, 8 ASICs with gaps, staggered quadrants; tile 15 = top of the bottom-right quadrant; edit `QUADS` in `agipd.py` to change) and numbered (mean image of each tile per scan, from the first `overview_trains` trains). Click tiles to choose the ones to analyse, then **Apply**.
  - **AGIPD selected tiles** in place of the second detector: the chosen tiles stacked top to bottom, one image per train (frames averaged per train), with the usual ROIs. ROIs stay on their tile when the choice changes.
  - **HIREX = pump, DES = probe**: each spectrum is `data.adc[a, b:c]`, where **a is the Gotthard frame within the train** (HIREX 20, DES 1, as the notebook comments: "frame 20, pixel window -> one pump spectrum / train") and b:c is the pixel window (1750:2250 and 1100:1250). Edge baseline (50 px per side) is subtracted, then the sum inside the draggable band is the pump / probe intensity used everywhere (cleaning, normalisation).
  - **Spectrometers & XGM page**: per spectrometer the mean spectrum with its band, the distribution and the value vs shot number, and the value vs XGM; the XGM distribution and vs shot; HIREX vs DES.
  - No delay motor: type the delay of each run in the scan table. No CRL page.
- **custom**: any proposal number; you choose the sources YAML.

Switching proposal rebuilds the window (the loaded scans are dropped).

## What it does

- **Two Jungfrau detectors** side by side. Each one shows its scan-average image: either the average of all used scans or a single scan.
  - **⤢ Full page** shows one Jungfrau across the whole window, which makes it easier to centre the ROIs. **⤡ Back** restores the layout and the zoom of each detector, and keeps every ROI edit, level and colour map change made in the meantime.
  - The **colour map** selector applies to both detectors. It offers viridis, inferno, magma, plasma, cividis and grey, plus maps where low counts are white (white → blue / viridis / red, and inverted grey). The default follows the theme: viridis in dark mode, white → blue in light mode. A map you pick yourself stays when you switch themes.
  - **log scale** shows both detectors as log10 (clipped at 10⁻³ of the max). It changes the display only; ROI sums stay linear.
- **ROIs on each detector.** They are elliptical by default. Each detector starts with four: three signal ROIs plus one background. You can:
  - click a ROI's **shape** cell to switch it between ellipse and rectangle; each ROI has its own shape;
  - add ROIs with **+ ROI** and remove them with **Delete**;
  - drag an ROI to move it, and drag its handles to resize or rotate it;
  - rename one by double-clicking its name;
  - switch one on or off;
  - flag one as **bkg**. A background ROI's mean level per pixel, times the ROI's pixel count, is subtracted from every other ROI on the same detector;
  - save or load each detector's ROI set as JSON.
- **Two diamond diodes** for the pump and probe intensities. These are used for shot cleaning (a per-scan MAD cut and a minimum fraction of the median) and for normalisation (none, probe, pump, pump × probe, or pump² × probe).
- **One delay per scan.** The GUI combines several scans into a delay series:
  - **Load** replaces the loaded scans, and **Add** appends to them;
  - the scan table lists each scan's delay, which is read from the delay source or typed in by hand;
  - a checkbox includes or excludes a scan;
  - scans at the same delay are merged.
  - runs can be typed in any order, with gaps and ranges (`92, 96, 94, 95`, `95 96 98-99`); the table can be ordered by delay, run number or as typed;
  - **run → delay file**: a delay typed in the table is saved at once to `delays_<proposal>.csv` in the start folder (`run, delay_fs` per line, editable by hand). When a run is loaded, its delay is filled in from that file (bold in the table); the file wins over a delay read from the data, and clearing the cell removes the run from the file.
- **Delay series page:**
  - the background-subtracted signal of each ROI vs delay;
  - the normalised signal (mean S / mean N, with a ratio-estimator error);
  - the mean pump and probe diode per scan.
- **Correlations page:**
  - pump vs probe;
  - pump and probe diodes, shown either as distributions (histograms) or as diode vs shot number. The **diodes** selector switches between the two. In the shot-number view, shots are in scan order, cut shots appear as grey ×, and dashed lines mark the scan boundaries;
  - the selected ROI vs pump, vs probe and vs the normaliser.
  - Points are coloured by delay, and the Pearson r is in each title.
- **Upstream spectrometers page** (toggle with **Spectrometers**, or `--no-spec`). It covers the spectrometers before the SDL; the diamond diodes sit after the SDL and the monochromators. For each spectrometer it shows:
  - the mean spectrum, with a draggable integration band;
  - a histogram of the per-shot integral (edge-baseline subtracted);
  - pump diode vs integral and probe diode vs integral, with Pearson r and a binned-mean line.
  - With self-seeded SASE the full integral includes the SASE pedestal, so r can be close to 0. Drag the band onto the seed line to integrate only that part.
- **CRL pointing page** (toggle with **CRL page**, or `--no-crl`). It shows the x/y position of each arm on the CRLs before the SDL, plus each arm's diode vs x and vs y with a binned-mean line. A feedback loop keeps the beams centred, so this page checks that the loop is keeping up.
- **Fitting page.** This page fits a delay-series ROI curve (normalised or raw, optionally scaled to max = 1) with the models from `GaO_Analysis_mac.ipynb`:
  - The model is S(t) = |½[1+erf((t−t0)/σ)] · (a_exp e^(−(t−t0)/τ) − Σ aᵢ sin(2πνᵢ(t−t0)+φᵢ) [e^(−(t−t0)/τᵢ)] + y0)|².
  - You choose **1, 2 or 3 oscillations**, whether each one is damped, and an optional 2nd harmonic of ν1. These choices cover freq2_dump1/2, freq3_dump1/2/3 and freq3_dump2_2nd_harm.
  - The parameter table holds the **start value, min, max and fix** for every parameter, plus the result ± error. Results that end at a bound are flagged.
  - Buttons: **Fit**, **Fit thermal only** (fit_exp), **Guess ν from FFT** (strongest peaks of the FFT of data − thermal; ν bounds set to ±20 %), **Results → start** and **Reset**. You can also set the fit range.
  - Plots: the data with the full fit and the thermal part; the thermal part alone (with τ); data − thermal and its FFT; the residual and its FFT. The fitted νᵢ are marked on both FFTs.
  - **Uneven delays:** a banner warns when the delays are not evenly spaced. Before every FFT, the points are linearly interpolated onto an even grid (median step, or a step you set). The plots show the measured (●) and interpolated (✕) points. Use "FFT from t ≥" to leave out early points, like the notebook's `cut`.
- **Parameter scan page** (toggle with **Param scan**). Plots the ROI signal against a parameter recorded per train: a motor stepped inside one run (sample rotation, grating–sample distance), a slow drift (temperature), or one value per run across several runs.
  - Type the **source** and **key** (or pick a preset read in the p7016 notebooks: sample rotation motor T3, motor T2, Cobra gas temperature), a name and unit, then **Read**. Only the parameter is read for the loaded runs; runs loaded later read it too. A wrong name does not lose the run: the page lists the runs without the parameter.
  - Shots of all used runs are pooled (cleaning and normaliser as elsewhere). **auto** gives one point per motor position when the parameter is stepped (≤ 60 positions), otherwise equal-width bins (number set by you); each point is mean S / mean N with its error.
  - Plots: normalised and raw signal of every ROI vs the parameter (or, for one ROI, one curve per run or per delay), kept shots coloured by delay, pump/probe vs the parameter, and the parameter vs shot number.
  - In a sources YAML: `param_enabled`, `param_label`, `param_unit`, `param: {source, key}` (Sources_7016.yaml has motor T3).
- **ROI position check** (button **ROI position**, opens its own window). For a chosen ROI: the **centre of mass** (counts minus the background-ROI level) or the **max position** (brightest pixel), x and y, either **shot by shot** or on the **mean image of each run** (one point per run, for several runs). Plots x and y vs shot / run and y vs x, coloured by delay, with the ROI centre marked.
- **Dark / light mode.** Switch with the button in the top right, or start with `--light`.
- **Export:** a CSV of the delay series. Its header holds the settings and the ROI geometry.

Screenshots (simulated data) are in `screenshots/`.

## Run

```bash
pip install -e . PyQt5 pyyaml scipy    # laptop
python -m xtg_mid_gui                                      # real data (extra_data), nothing loaded: choose scans, press Load
python -m xtg_mid_gui --proposal 7016 --scans 90-100       # load these p7016 runs at start (Maxwell)
python -m xtg_mid_gui --backend simulated --scans 101-149  # fake data for testing (uneven delays, shuffled order)
python -m xtg_mid_gui --backend simulated --scans 101-149 --preset 7016 --light
```

On Maxwell, inside the proposal `usr/` folder:

```bash
bash run_xtg_mid_gui.sh   # uses `module load exfel exfel-python`, builds ../xtg_mid_env on first use
```

It works the same way as `xtg_gui/run_xtg_gui.sh`: the venv is built with `--system-site-packages`, and only pyqtgraph gets pip-installed.

## Plugging in the real names

1. `python -m xtg_mid_gui --write-config sources.yaml`. (`sources.example.yaml` is a copy of this output.)
2. Replace every `PLACEHOLDER_...` source and key, and set `proposal`. For each entry:
   - `pulses: "a:b"` and `reduce: mean|sum` choose how per-pulse or per-cell data is reduced to one value per train;
   - `scale` and `offset` convert to physical units. For example, set them on `delay` to turn a motor position into fs.
3. Save it as `Sources_11456.yaml` in the folder you start the GUI from and choose proposal 11456 (or start with `--config sources.yaml --backend extra_data --scans 20-40`).

The header shows how many source names are still placeholders. **Sources…** lists them all.

## Code layout

| file | role |
|---|---|
| `config.py` | `SourceConfig`. It is the only place where source names appear. It loads and saves YAML or JSON. |
| `delays.py` | The run → delay file (`delays_<proposal>.csv`). |
| `agipd.py` | AGIPD1M tile layout (all-tiles canvas, tile numbers, stacked analysis tiles). |
| `roicheck.py` | ROI position check window (centre of mass / max). |
| `presets/` | `Sources_7016.yaml` and the `Sources_11456.yaml` template. |
| `backends.py` | `ScanData`, `SimulatedBackend` (self-seeded SASE toy model) and `ExtraDataBackend` (reads processed runs via `extra_data`; one scan = one run). To add another backend, implement `load_scan(scan) -> ScanData`. |
| `fitting.py` | Qt-free fit models, thermal part, bounded `curve_fit`, FFT on an even grid, and ν guess (ported from `GaO_Analysis_mac.ipynb`). |
| `fitpage.py` | The Fitting page. |
| `analysis.py` | Qt-free maths: ellipse masks, ROI sums with background subtraction, the MAD cut, normalisers, the ratio of means, combining scans into a delay series, and Pearson r. |
| `rois.py` | Elliptical ROI objects and the per-detector ROI table. |
| `theme.py` | Dark and light palettes, the stylesheet, and live restyling of the plots. |
| `gui.py` | The main window and its pages. |

## Tests

`python -m pytest tests`: 59 tests, covering the analysis maths (including spectrometer integrals), the config round-trip, the simulated backend (including self-seeded spectra), and an offscreen GUI smoke test (the fit models and FFTs on uneven delays, the Fitting page; load; add, rename, move, reshape and delete ROIs; append scans; theme and colour maps; log scale; full-page view; diode views; CRL, spectrometer and parameter-scan pages; CSV export).

## Not done yet / to confirm

- The real source names and keys for the Jungfraus, diodes, delay, upstream spectrometers (and their pixel window / eV calibration) and the CRL position monitors.
- `ExtraDataBackend` has not been run on real files, because the names are unknown and this was built without Maxwell access. Test it on the first processed run.
- Jungfrau geometry, masks, and per-cell handling beyond a mean over cells.
- Memory: every per-shot image of every loaded scan is kept in RAM. Use `max_trains` in the config for long scans.
