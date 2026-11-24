# Archived full-run results

These files are the small tables and representative figures from the completed analysis. The course report is generated from the full ledgers under `outputs/run/`. The accompanying project archive includes those ledgers and execution logs. An independent rerun will regenerate them from the [versioned MIT-BIH data](https://doi.org/10.13026/C2F305).

- `summary.csv` gives the baseline, reference-location, matched and end-to-end protocol-level results.
- `noise_metrics.csv` gives all nine paired comparisons, including the subject-bootstrap percentile intervals.
- `per_class.csv` and `confusion.csv` give class metrics and the raw confusion counts. The sentinel `none` represents a missing detection or an extra prediction.
- `paired_detection.csv` gives the same-beat and end-to-end contrasts. `detection_by_record.csv` includes the pooled row `ALL`.
- `beat_counts.csv` provides the cohorts and their label composition. The source for both figures is available in the accompanying project archive.
- `sensitivity_record_213.csv` gives the full primary and post hoc reduced-cohort metrics at reference positions without added noise.
- `minority_errors.csv` gives S classification at reference positions separately from N and V misses in the detector experiment without added noise.
- `stress_metrics.csv` and `stress_paired_delta.csv` give the additional 0 dB drift XQRS results. The recorded waveform receives the same drift in both branches. The end-to-end recall denominators include missed beats.
