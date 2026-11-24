# How Much Does Filtering Help?

This project tests whether a 0.5–40 Hz band-pass helps classify MIT-BIH heartbeats from people absent from training. The two branches are **No external band-pass** and **With external band-pass**. Both subtract a local baseline when calculating morphology features. The noise experiment uses reference beat positions. The primary detector experiment uses XQRS on recordings without added noise.

## Results

The full test uses 47,552 N/S/V/F beats from 21 held-out subjects. The paired intervals resample subjects with both models and the injected waveforms fixed.

| Experiment at reference positions | No external band-pass macro-F1 | With external band-pass macro-F1 | Difference |
| --- | ---: | ---: | ---: |
| No added noise | 0.5495 | 0.5237 | −0.0258 |
| White Gaussian noise, 0 dB | 0.4954 | 0.4835 | −0.0119 |
| 0.33 Hz baseline drift, 0 dB | 0.5057 | 0.5234 | +0.0177 |

The paired 95% intervals include zero in all nine conditions. On the original recordings, XQRS reached a detection F1 of 0.9978. With the external band-pass, the classification macro-F1 changed from 0.5237 at reference positions to 0.5183 with detection errors included. The post hoc record 213 comparison and the separate 0 dB drift replay are reported in the result tables. [Result tables](results/)

![Macro-F1 under two types of noise, with paired subject intervals for filtered minus raw](results/fig_robustness.png)

## Reproduce on CPU

Use Linux or WSL with Python 3.12 and uv. Unpack the project archive or open the repository checkout, then run these commands from the project root. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked
uv run pytest -q
uv run python -m bmsp_ecg data fetch --config configs/experiment.yaml
uv run python -m bmsp_ecg smoke --config configs/experiment.yaml
uv run python -m bmsp_ecg run --config configs/experiment.yaml
uv run python -m bmsp_ecg stress --config configs/experiment.yaml
uv run python scripts/verify_results.py outputs/run
uv run python -m bmsp_ecg report --config configs/experiment.yaml --draft
```

The data command downloads the versioned WFDB header, signal and reference files over HTTPS and compares them to the official SHA-256 list. Valid existing files are reused. With an existing copy, append `--data-dir /path/to/mitdb` to the data, smoke and run commands. `data check` confirms triplets exist and checks the official digests when `SHA256SUMS.txt` is present.

The smoke run uses six real recordings and 12 trees per branch. The complete run uses 300 trees per branch. The measured full experiment was 180.5 s with peak process RSS of 812 MiB on Linux x86-64, CPython 3.12.14 and two CPU workers. It excludes dependency setup, source data checks and PDF rendering. See [verification notes](docs/VERIFICATION.md).

The stress command reuses the two models from `run`. It injects the same 0 dB drift into the same test cohort, then runs XQRS separately for each branch. The command below rebuilds the sensitivity and minority-error tables from the two saved ledgers and the saved run configuration.

```bash
uv run python -m bmsp_ecg analyze --config configs/experiment.yaml
```

Pass the same `--output-dir` to commands that share a run. If the reports cannot find the full run, execute `run` first. For a Python version mismatch, use `uv sync --locked --python 3.12`. Re-run the fetch command to repair a file that fails the hash check.

## Experiment design

- **Acquisition and split.** 360 Hz, MLII chosen by header name. Exclude paced records 102, 104, 107 and 217. The modified DS1/DS2 puts records 201 and 202 in the same training subject group. The fixed manifest is [configs/split.csv](configs/split.csv).
- **Models.** Window [−0.25, +0.35) s, RR context and fixed-time morphology features, training-only median imputation and class-weighted random forests. The filtered arm applies a full-record forward–backward 0.5–40 Hz Butterworth filter. The configuration is [configs/experiment.yaml](configs/experiment.yaml).
- **Noise.** White Gaussian noise and 0.33 Hz drift are injected at 24, 12, 6 and 0 dB relative to the demeaned recorded signal. Both arms see the same waveform realization per record and condition.
- **Evaluation.** Treat N/S/V/F equally in macro-F1. Evaluate noise at reference positions and score the primary XQRS experiment without added noise. The 0 dB baseline drift replay scores QRS F1, end-to-end macro-F1 and S/V recall in both branches. XQRS uses its internal processing in each branch. Zero-phase filtering and following RR intervals use future samples.

The detector call uses `xqrs_detect(sig=filtered, fs=360, conf=None, learn=True, verbose=False)`. WFDB 4.3.1 defaults to an initial heart rate of 75 beats/min, a 25–200 beats/min range, 0.10 s expected QRS width, 0.20 s refractory period, thresholds of 0.13 initially and 0 at minimum, and a 0 s T-wave inspection period. The evaluation match tolerance is 150 ms.

## Output files

| Path under the run directory | Contents |
| --- | --- |
| `data_audit.csv`, `beat_counts.csv`, `training_beats.csv.gz` | File checksums, lead columns, symbol/beat counts and training index |
| `reference_predictions.csv.gz`, `detection_events.csv.gz` | Per-beat classification and detection ledgers |
| `summary.csv`, `per_class.csv`, `confusion.csv` | Protocol, class and confusion statistics |
| `noise_metrics.csv`, `snr_audit.csv`, `paired_detection.csv` | Robustness and detection contrasts, calibration and bootstrap intervals |
| `sensitivity_record_213.csv`, `minority_errors.csv` | Post hoc sensitivity and classification/detection error counts |
| `stress/events.csv.gz`, `stress/metrics.csv`, `stress/paired_delta.csv` | Drift replay QRS and classification results |
| `stress/noise_audit.csv`, `stress/provenance.json` | Shared waveform hashes, SNR and source run/model hashes |
| `detection_by_record.csv`, `detection_window_skips.csv` | Per-record QRS metrics and clipped edge counts |
| `qc_*`, `failure_*`, `figures/` | ECG figure source data and generated vector/PNG graphics |
| `run_metadata.json`, `*_model.joblib` | Timing, memory, environment and fitted pipelines |

Smoke writes to `outputs/smoke/`, the complete run to `outputs/run/`, the stress replay to `outputs/run/stress/` and the documents to `submission/`. Load Joblib models only from a trusted run.

## Build the final PDFs

Save the cover fields in a local `identity.yaml`. Use the optional `name_native` field for a second spelling.

```yaml
name: "Your name as it should appear on the cover"
name_native: "Optional second spelling"
student_id: "Your student ID"
date: "YYYY-MM-DD"
repository: "https://github.com/OWNER/REPOSITORY"
```

```bash
uv run python -m bmsp_ecg report --config configs/experiment.yaml --identity identity.yaml
```

The renderer writes `BMSP_Report.pdf` for **1. report** and `BMSP_Code_Documentation.pdf` for **2. Code**. The submission directory and local identity file are ignored by Git.

## Sources

Data, method references and workflow attribution appear in [docs/SOURCES.md](docs/SOURCES.md). This educational experiment measures offline beat classification on one historical dataset.
