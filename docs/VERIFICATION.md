# Verification record

Verified on 2026-11-24 with CPython 3.12.14, uv 0.13.9 and Linux x86-64. The runtime exposed two CPU equivalents and an 8 GiB memory limit. Raw files came from the official MIT-BIH v1.0.0 distribution.

| Stage | Command or check | Exit | Evidence |
| --- | --- | ---: | --- |
| 0 · Environment | `uv sync --locked` and `uv run python -m bmsp_ecg --help` | 0 | Lock resolved 46 packages and installed 45 in the Python 3.12 environment. |
| 0 · Empty data | `uv run python -m bmsp_ecg data check --data-dir /tmp/bmsp-no-such-db` | 2, expected | The error identifies missing record files and the fetch or data-directory options. |
| 1 · Acquisition | `uv run python -m bmsp_ecg data fetch --config configs/experiment.yaml --data-dir /tmp/bmsp-verified-fetch` | 0 | 48 triplets and 144 official SHA-256 matches in the 12.0 s clean download. |
| 1–3 · Protocol and features | `uv run pytest -q` | 0 | 10 tests pass. Includes disjoint subject groups, annotation mapping, physical lead lookup, window edges, filter response, calibrated SNR and exact QRS matching. |
| 2–4 · Real-input smoke | `uv run python -m bmsp_ecg smoke --config configs/experiment.yaml` | 0 | Six recordings, 6,692 training beats and 8,121 held-out labeled beats. Output under `outputs/smoke/`. |
| 3–4 · Fixed full run | `uv run python -m bmsp_ecg run --config configs/experiment.yaml` | 0 | 53,132 training beats, 47,552 held-out labeled beats, 180.5 s experiment time and 812 MiB peak RSS. Output under `outputs/run/`. |
| 4 · Recompute metrics | `uv run python scripts/verify_results.py outputs/run` | 0 | 903,488 classifications, 47,616 detection rows, nine noise settings and all reported metric tables recomputed from the event ledgers. |
| 5–6 · Reports | `uv run python -m bmsp_ecg report --config configs/experiment.yaml --draft` | 0 | 7-page course report and 3-page code guide, with searchable text and vector result figures. All pages were rendered for visual review. |
| 7 · Clean replay | Copy source and lock into a fresh directory, then run locked install, tests, data check, smoke, full run and draft report | 0 | Full replay finished in 195.6 s. The fresh environment used an independently downloaded, checksum-verified dataset. |
| 7 · Compare independent runs | `uv run python scripts/verify_results.py outputs/run --compare /path/to/replay/outputs/run` | 0 | 18 files matched byte-for-byte, including training indices, both prediction ledgers, all public metric tables and the two waveform data extracts. |

The archive includes the original run's prediction ledgers, source tables and execution logs. The result comparison excludes elapsed time, RSS and PDF metadata because those are run-specific.
