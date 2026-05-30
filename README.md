# Grade-Then-Screen: Measurement-Trustworthiness Grading Before Vibration Trend Screening

Analysis code for the framework described in the paper *"Measurement-Trustworthiness Grading Before Trend Screening: A Vibration Condition-Monitoring Framework for Electromechanical Transmission Components"* (submitted to the *Journal of Vibration Engineering & Technologies*).

The framework grades every acceleration record for **measurement trustworthiness** — against the sensor range and the acquisition-chain full scale — *before* any health indicator is computed, then screens per-session RMS trends with multiplicity-corrected statistics and several ground-truth-free realism checks.

## Pipeline / scripts

| Script | Purpose |
|---|---|
| `vibe_core.py` | Core binary decoding and feature extraction (time / frequency / envelope) |
| `ws1_4_methods.py` | Trend screening: Benjamini–Hochberg-corrected Spearman, Mann–Kendall, Pearson, Theil–Sen, leave-one-out |
| `ws5_autocorr_ci.py` | Residual-autocorrelation diagnostics and confidence intervals |
| `ws_threshold_sensitivity.py` | Sensitivity of the grading to the electrical-spike threshold |
| `ws_r2_spatial.py` | Cross-sensor spatial-consistency corroboration |
| `ws_r3_trajmodel.py` | Trajectory modelling (linear/exponential, AIC) and change-point |
| `ws_r4_order.py` | Order / spectral attribution against shaft-order families |
| `ws_raw_trajectory.py` | In-record sliding-window RMS/kurtosis trajectories |
| `pub_figures_jvet.py`, `pub_fig4_inrecord.py` | Publication figure generation |

## Data layout

The scripts expect:

- `./data/features_all.parquet` — per-record feature table (input to most scripts);
- `./data/raw/` — raw acceleration records, only for the whole-waveform steps;

and write outputs to `./results/`. The dataset is available from the authors on reasonable request, subject to institutional approval. Set these paths at the top of each script or via the corresponding variables.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `pandas`, `matplotlib`, and `pyarrow`.

## License

Released under the MIT License — see `LICENSE`.
