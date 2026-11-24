"""Generate the English research report and reproducibility guide."""

import json
import re
from datetime import date
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import sosfreqz

from .figures import build_figures
from .layout import (INK, TEAL, Spacer, VectorFigure, build_pdf, callout, code,
                     heading, note, para, register_fonts, rule, section, table)
from .signal import CLASSES, filter_sos

TITLE = "How Much Does Filtering Help?"
SUBTITLE = "Noise-Robust ECG Beat Classification on Unseen Subjects"
REPORT = "BMSP_Report.pdf"
GUIDE = "BMSP_Code_Documentation.pdf"


def f1(value):
    return f"{value:.4f}"


def signed(value):
    return f"{value:+.4f}"


def pct(value):
    return f"{100 * value:.2f}%"


def number(value):
    return f"{int(value):,}"


def ci(row):
    return f"[{signed(row.lo)}, {signed(row.hi)}]"


def link(url, label=None):
    return f'<link href="{escape(url, quote=True)}" color="{TEAL}">{escape(label or url)}</link>'


def load_identity(path, draft):
    value = yaml.safe_load(path.read_text()) if path else {}
    value = value or {}
    if not isinstance(value, dict):
        raise ValueError("Identity YAML must contain a mapping")
    fields = ("name", "student_id", "date", "repository")
    identity = {key: str(value[key]).strip() if value.get(key) is not None else "" for key in fields}
    if not draft and any(not identity[key] for key in fields):
        raise ValueError("Formal PDFs require --identity YAML with name, student_id, date and repository. "
                         "Use --draft for review copies.")
    if identity["date"]:
        date.fromisoformat(identity["date"])
    if identity["repository"] and not re.fullmatch(
        r"https://github\.com/[\w.-]+/[\w.-]+/?", identity["repository"]
    ):
        raise ValueError("repository must be a public https://github.com/OWNER/REPO URL")
    return identity


def read_results(run):
    return {
        "meta": json.loads((run / "run_metadata.json").read_text()),
        "case": json.loads((run / "failure_meta.json").read_text()),
        "scores": pd.read_csv(run / "summary.csv").set_index("protocol"),
        "noise": pd.read_csv(run / "noise_metrics.csv", dtype={"snr_db": str}, keep_default_na=False),
        "counts": pd.read_csv(run / "beat_counts.csv"),
        "audit": pd.read_csv(run / "data_audit.csv", dtype={"record_id": str}),
        "class": pd.read_csv(run / "per_class.csv"),
        "matrix": pd.read_csv(run / "confusion.csv", keep_default_na=False),
        "detection": pd.read_csv(run / "detection_by_record.csv", dtype={"record_id": str}).set_index("record_id"),
        "paired": pd.read_csv(run / "paired_detection.csv").set_index("comparison"),
        "snr": pd.read_csv(run / "snr_audit.csv"),
    }


def cover(identity, results):
    n = results["meta"]
    repo = identity["repository"]
    rows = [
        ("Student", escape(identity["name"] or "Pending confirmation")),
        ("Student ID", escape(identity["student_id"] or "Pending confirmation")),
        ("Date", escape(identity["date"] or date.today().isoformat())),
        ("Public repository", link(repo) if repo else "Pending publication"),
    ]
    return [
        Spacer(1, 60),
        para("Course project report", size=12, leading=15, face="BodyBold", color=TEAL, after=21),
        para(TITLE, size=38, leading=44, face="SerifBold", color=INK, after=12),
        para("Noise-Robust ECG Beat Classification<br/>on Unseen Subjects", size=19, leading=26, face="Serif", color=INK, after=28),
        rule(), Spacer(1, 21),
        para("Biomedical Signal Processing · Fall 2026", size=12, leading=16, face="BodyBold", after=10),
        note("Dr. DUAN Fang · Huaqiao University"),
        *[para(f"<b>{label}</b> &nbsp; {value}", size=10.5, leading=15.5, after=8)
          for label, value in rows],
        Spacer(1, 26),
        callout("RESEARCH QUESTION",
                "How does a 0.5–40 Hz band-pass change beat classification under controlled noise, "
                "and how much does performance change when an automatic detector supplies the beat locations?"),
        Spacer(1, 22),
        note(f"MIT-BIH Arrhythmia Database v1.0.0 · single-lead MLII · "
             f"{number(n['test_subjects'])} held-out subjects"),
    ]


def acquisition(results, figure):
    counts = results["counts"].groupby("partition").sum(numeric_only=True)
    train, test = counts.loc["train"], counts.loc["test"]
    n = results["meta"]
    missing = results["audit"].query("status == 'missing_mlii'").record_id.tolist()
    columns = results["audit"].query("status == 'included' and mlii_column_zero_based == 1").record_id.tolist()
    lead_note = ("Included records without MLII: " + ", ".join(missing) + ".") if missing else (
        "MLII was available in all 44 eligible records.")
    return [
        section("Acquisition and study design"),
        para("MIT-BIH v1.0.0 contains 48 approximately half-hour, two-channel ambulatory recordings "
             "from 47 individuals. The signals were digitized at 360 Hz per channel with 11-bit resolution "
             "over a 10 mV range. Independent cardiologist annotations were reconciled to create the "
             "reference beat locations and types [1, 2]."),
        para("WFDB decoded the physical signal in mV. The pipeline selected MLII by header name in each "
             f"record. Record {', '.join(columns)} has MLII in column 1 (zero-based). {lead_note} "
             f"The data audit records annotation symbols, selected columns and {n['official_hashes']} verified "
             "official file hashes."),
        heading("Subject-grouped modified DS1/DS2"),
        para("Records 102, 104, 107 and 217 were excluded because they contain paced beats. Record 202 "
             "moved from the conventional DS2 to the training side because it shares a subject with "
             "record 201 [1, 3]. The manifest verifies that no subject group appears in both partitions."),
        table(["Partition", "Records", "Subjects", "N", "S", "V", "F", "N/S/V/F total"],
              [["Training", n["training_records"], n["training_subjects"], *[number(train[x]) for x in CLASSES],
                number(sum(train[x] for x in CLASSES))],
               ["Held-out test", n["test_subjects"], n["test_subjects"], *[number(test[x]) for x in CLASSES],
                number(sum(test[x] for x in CLASSES))]],
              [.16, .115, .125, .115, .095, .10, .09, .20]),
        Spacer(1, 8),
        note(f"Table 1. Counts after full-window eligibility. Train/test each contributed "
             f"{number(train.Q)}/{number(test.Q)} Q beats and {number(train.unmapped_qrs)}/"
             f"{number(test.unmapped_qrs)} other QRS beats excluded from classification. "
             f"Window boundaries excluded {number(train.qrs_without_window)}/{number(test.qrs_without_window)} "
             f"QRS references. Non-QRS events totaled {number(train.non_qrs_events)}/{number(test.non_qrs_events)}."),
        heading("Signal quality check"),
        figure("qc"),
        note("Figure 1. A fixed three-second segment from training record 114. Both panels share the "
             "same amplitude scale. Dashed lines mark reference beat times."),
    ]


def methodology(spec):
    fs = spec["dataset"]["fs_hz"]
    frequencies = [0.33, 0.5, 10, 40, 75]
    _, response = sosfreqz(filter_sos(fs, spec["filter"]), worN=frequencies, fs=fs)
    gains = np.round(40 * np.log10(np.abs(response)), 1) + 0.0
    return [
        section("Methodology"),
        heading("Preprocessing and beat representation"),
        para("The raw branch uses the original MLII samples. The filtered branch applies a 0.5–40 Hz "
             "Butterworth band-pass to the full record using second-order sections and forward–backward "
             "filtering [4]. A fourth-order prototype yields four biquads in the single-pass band-pass. "
             f"The effective response has zero phase and squared magnitude. Its calculated amplitude "
             f"response at 0.33, 0.5, 10, 40 and 75 Hz is {', '.join(f'{g:.1f}' for g in gains)} dB, respectively."),
        para("Each beat contributes a half-open window from −250 to +350 ms (216 samples). Morphology "
             "features use the median from −200 to −120 ms as a local baseline. Both branches use the "
             "same 28 feature definitions."),
        table(["Feature group", "Definition and units"], [
            ["RR (4)", "Previous/next interval in seconds and each interval divided by its centered "
             "11-interval running median. RR uses all recognized QRS positions as timing context, including "
             "positions outside the feature window boundary."],
            ["QRS (8)", "Center amplitude (mV). In −80 to +100 ms, compute min, max, peak-to-peak and RMS "
             "(mV), absolute area (mV·s), duration at or above half the absolute peak (s), and maximum "
             "absolute first difference per second (mV/s)."],
            ["Whole beat (2)", "RMS and peak-to-peak amplitude (mV)."],
            ["Binned shape (14)", "Mean amplitude (mV) in consecutive 30 ms bins from −180 to +240 ms."],
        ], [.23, .77]),
        Spacer(1, 8),
        note("Table 2. Time bounds are relative to the supplied center and are rounded to the nearest "
             "sample. The local median uses nearest-edge padding. A missing neighbor gives a missing RR feature."),
        heading("Labels and model fitting"),
        para("The four-class mapping follows the morphology and interval study [3]. N = N/L/R/e/j, "
             "S = A/a/J/S, V = V/E and F = F. Q contains /, f and Q. WFDB QRS symbols B, n and r "
             "are tracked as unmapped. Reference QRS centers are used for fitting."),
        para("Each branch fits a separate 300-tree random forest to the same unperturbed training beats "
             "with balanced class weights per bootstrap sample, square-root feature sampling, two samples "
             "per leaf minimum and seed 877. Median imputation is fitted inside each training pipeline. "
             "The configuration uses fixed hyperparameters without a tuning stage."),
        callout("OFFLINE SCOPE", "Both zero-phase filtering and following RR intervals use future samples. "
                "The evaluation concerns offline beat classification on held-out subjects."),
    ]


def noise_results(results, figure):
    scores = results["scores"]
    test = scores.loc["reference_raw"]
    base = scores.loc["reference_all_n"]
    filt = scores.loc["reference_filtered"]
    noise = results["noise"]
    extrema = [noise.query("noise == 'clean'").iloc[0],
               noise.query("noise == 'white' and snr_db == '0'").iloc[0],
               noise.query("noise == 'drift' and snr_db == '0'").iloc[0]]
    discrepancy = (results["snr"].target_db - results["snr"].actual_db).abs().max()
    drift_levels = noise.query("noise == 'drift'").filtered_f1
    crossing = ((noise.lo <= 0) & (noise.hi >= 0)).sum()
    raw_f = results["class"].query("protocol == 'reference_raw' and label == 'F'").iloc[0].f1
    filtered_f = results["class"].query("protocol == 'reference_filtered' and label == 'F'").iloc[0].f1
    return [
        section("Results · Filtering under noise"),
        para("E1 and E2 use reference beat locations in both model branches. For E2, a record-specific "
             "zero-mean white Gaussian draw or a 0.33 Hz sinusoid with seeded phase was added at "
             "24, 12, 6 and 0 dB. Amplitude was scaled by "
             "√(P<sub>signal</sub> / (P<sub>noise</sub> × 10<super>SNR/10</super>)). "
             "Signal power used the demeaned original. Each pair saw the same perturbed recording, "
             f"and recorded SNR matched the target to within {max(discrepancy, 1e-9):.0e} dB."),
        para(f"With no added noise, always-N achieved {pct(base.accuracy)} accuracy and {f1(base.macro_f1)} "
             f"macro-F1. The raw and filtered forests scored {f1(test.macro_f1)} and {f1(filt.macro_f1)}, respectively."),
        figure("robustness"),
        note(f"Figure 2. The x-axis shows ordered settings. Each setting uses the same {number(test.support)} labeled beats from "
             f"{number(results['meta']['test_subjects'])} test subjects. Δ is filtered minus raw. Intervals are "
             "2.5th–97.5th percentiles from 2,000 paired test-subject resamples with models fixed. "
             "Class F1 uses zero for a zero denominator. At each SNR the signal reference is the recorded, "
             "already noisy ECG."),
        table(["Selected condition", "Raw F1", "Filtered F1", "Δ F1", "95% paired interval"], [
            [name, f1(row.raw_f1), f1(row.filtered_f1), signed(row.delta), ci(row)]
            for name, row in zip(["No added noise", "White noise · 0 dB", "Baseline drift · 0 dB"], extrema)
        ], [.27, .135, .155, .13, .31]),
        Spacer(1, 7),
        note("Table 3. Selected endpoints from the complete noise grid in Figure 2."),
        para(f"All {crossing} of {len(noise)} condition-level intervals include zero. Under added drift, "
             f"the filtered scores stayed within {f1(drift_levels.min())}–{f1(drift_levels.max())}. "
             f"On nominal recordings, fusion F1 fell from {f1(raw_f)} to {f1(filtered_f)} after filtering. "
             "That class accounted for the largest share of the reduction in clean macro-F1."),
    ]


def detection_results(results):
    scores, d = results["scores"], results["detection"].loc["ALL"]
    paired = results["paired"]
    cond = paired.loc["same_beats_reference_to_matched"]
    e2e = paired.loc["reference_to_end_to_end"]
    base = results["class"].query("protocol == 'reference_filtered'").set_index("label")
    end = results["class"].query("protocol == 'end_to_end'").set_index("label")
    raw = results["class"].query("protocol == 'reference_raw'").set_index("label")
    constant = results["class"].query("protocol == 'reference_all_n'").set_index("label")
    groupings = [
        ("reference_filtered", "Reference positions", "All labeled beats"),
        ("reference_on_matches", "Reference positions", "Same matched beats"),
        ("matched_classification", "XQRS positions", "Same matched beats"),
        ("end_to_end", "XQRS end to end", "All labeled beats + extra detections"),
    ]
    return [
        section("Results · From QRS to class"),
        para("E3 used the filtered model and the recordings without extra noise. WFDB 4.3.1 XQRS ran "
             "with its default configuration and signal-based initialization enabled. XQRS applies its own "
             "5–20 Hz band-pass internally [5]. All automatic RR and waveform features came from detected "
             "positions. Predictions and reference beats were evaluated in the same full-window interval."),
        para("A one-to-one matcher within ±150 ms (54 samples) first maximized pair count and then minimized "
             "total absolute sample error. All recognized QRS types were eligible for detection matching. "
             "A missed included beat adds a false negative, an extra detection adds a false positive for "
             "its predicted class, and a wrong class adds one of each. Matches to Q or unmapped beats "
             "are counted separately."),
        callout("QRS DETECTION",
                f"Precision {pct(d.precision)} · recall {pct(d.recall)} · F1 {f1(d.f1)}<br/>"
                f"{number(d.tp)} matches · {number(d.fn)} missed references · {number(d.fp)} extra "
                f"detections · {number(d.matched_excluded)} matches to excluded classes"),
        Spacer(1, 11),
        table(["Position source", "Evaluation support", "N/S/V/F refs", "Macro-F1"], [
            [title, subset, number(scores.loc[key].support), f1(scores.loc[key].macro_f1)]
            for key, title, subset in groupings
        ], [.23, .46, .16, .15]),
        Spacer(1, 7),
        note("Table 4. The middle two rows compare identical beats and isolate the change in center and "
             "detector-derived RR among matches. The final row includes unmatched detection errors."),
        para(f"Median absolute matched offset was {d.median_abs_ms:.1f} ms and the 95th percentile was "
             f"{d.p95_abs_ms:.1f} ms. On matched beats the switch to automatic positions changed "
             f"macro-F1 by {signed(cond.delta)} (95% interval {ci(cond)}). Overall end-to-end change "
             f"relative to all reference centers was {signed(e2e.delta)} (95% interval {ci(e2e)})."),
        table(["Class", "Test beats", "Always-N recall", "Raw recall", "Filtered recall",
               "Filtered precision", "Filtered F1", "End-to-end F1"],
              [[label, number(base.loc[label].support), pct(constant.loc[label].recall),
                pct(raw.loc[label].recall), pct(base.loc[label].recall),
                pct(base.loc[label].precision), f1(base.loc[label].f1), f1(end.loc[label].f1)]
               for label in CLASSES], [.09, .115, .145, .125, .125, .14, .12, .14]),
        Spacer(1, 7),
        note("Table 5. Per-class results on recordings without added noise. The first three models use reference centers. N combines normal and similar "
             "beats, S is supraventricular ectopic, V is ventricular ectopic, and F is fusion."),
    ]


def discussion(results, figure):
    case = results["case"]
    label = case["focus_label"]
    t0, t1, centre = (case[key] / case["fs"] for key in ("start", "stop", "focus_sample"))
    condition = ("A missed reference beat" if case["focus_status"] == "miss" else
                 f"A reference beat predicted as {case['focus_prediction']}")
    confusion = results["matrix"].query("protocol == 'end_to_end'").pivot(index="true", columns="pred", values="count")
    base = results["class"].query("protocol == 'reference_filtered'").set_index("label")
    return [
        section("Discussion"),
        heading("A missed ventricular beat"),
        figure("failure"),
        note(f"Figure 3. {condition} ({label}) at {centre:.3f} s in test record "
             f"{case['record_id']}, sample {case['focus_sample']:,}. The segment covers "
             f"[{t0:.3f}, {t1:.3f}) s. Dashed lines show references, triangles show XQRS detections and "
             "shading marks the matching window. The case rule chooses the earliest eligible V miss "
             "in the lowest numbered test record, at least two seconds from either edge."),
        para("A missed center removes one classification opportunity and lengthens the interval between "
             "neighboring detected beats. The same-beat comparison therefore captures shifts in both "
             "the morphology window and the measured RR context. The example shows the event accounted "
             "for by an end-to-end false negative. The trace alone cannot identify the detector's internal cause."),
        table(["Reference / prediction", "N", "S", "V", "F", "No detection"],
              [[label if label != "none" else "Extra detection",
                *[number(confusion.loc[label, pred]) for pred in (*CLASSES, "none")]]
               for label in (*CLASSES, "none")], [.25, .145, .12, .12, .12, .245]),
        Spacer(1, 7),
        note("Table 6. End-to-end confusion counts on the filtered branch. Q and other excluded-class "
             "matches are accounted for in the QRS tally. The final column and row retain misses and extra detections."),
        heading("Timing, morphology and class imbalance"),
        para(f"At reference centers, filtered V recall was {pct(base.loc['V'].recall)}, compared with "
             f"{pct(base.loc['S'].recall)} for S and {pct(base.loc['F'].recall)} for F. RR captures premature "
             "timing while QRS features capture changes in shape, width and slope. A supraventricular "
             "ectopic beat may preserve a narrow QRS, and a fusion beat may resemble either dominant "
             "morphology. These mechanisms are consistent with overlapping feature distributions, "
             "which were not formally isolated here."),
    ]


REFERENCES = [
    ("Moody G, Mark R. MIT-BIH Arrhythmia Database v1.0.0. PhysioNet, 2005.", "https://doi.org/10.13026/C2F305"),
    ("Moody GB, Mark RG. The impact of the MIT-BIH arrhythmia database. IEEE Eng Med Biol Mag. 2001;20(3), 45–50.",
     "https://doi.org/10.1109/51.932724"),
    ("de Chazal P, O’Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology "
     "and heartbeat interval features. IEEE Trans Biomed Eng. 2004;51(7), 1196–1206.",
     "https://doi.org/10.1109/TBME.2004.827359"),
    ("SciPy documentation. Forward–backward filtering with second-order sections.", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html"),
    ("WFDB Python documentation. XQRS and ECG signal processing.", "https://wfdb.readthedocs.io/en/latest/processing.html"),
    ("Moody G, Muldrow W, Mark R. MIT-BIH Noise Stress Test Database v1.0.0. PhysioNet.",
     "https://physionet.org/content/nstdb/1.0.0/"),
    ("Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills. arXiv, 2026.",
     "https://doi.org/10.48550/arXiv.2609.00065"),
]


def conclusion(results):
    scores, noise, counts = (results[key] for key in ("scores", "noise", "counts"))
    white = noise.query("noise == 'white' and snr_db == '0'").iloc[0]
    drift = noise.query("noise == 'drift' and snr_db == '0'").iloc[0]
    test = counts.query("partition == 'test'")
    concentration = test.sort_values("F", ascending=False).iloc[0]
    e2e = results["paired"].loc["reference_to_end_to_end"]
    matrix = results["matrix"].query("protocol == 'end_to_end'")
    missed_v = matrix.query("true == 'V' and pred == 'none'").iloc[0]["count"]
    return [
        section("Conclusion"),
        para(f"On {number(results['meta']['test_subjects'])} held-out subjects, the reference-centered "
             f"forests scored {f1(scores.loc['reference_raw'].macro_f1)} raw and "
             f"{f1(scores.loc['reference_filtered'].macro_f1)} filtered without added noise. The filtered-minus-raw "
             f"difference at 0 dB was {signed(white.delta)} for white noise and {signed(drift.delta)} "
             "for 0.33 Hz drift. The results give no consistent advantage to external filtering "
             "across the tested noise conditions. With automatic QRS positions and detection errors included, "
             f"the filtered pipeline achieved {f1(scores.loc['end_to_end'].macro_f1)} macro-F1."),
        heading("Interpretation and limits"),
        para("A Butterworth band-pass preferentially rejects energy outside the passband. Narrowband "
             "0.33 Hz drift is strongly attenuated, and the common local baseline subtraction can also "
             "attenuate slow offsets in the raw branch. White noise has energy both inside and outside "
             "the passband, so QRS-related slopes and shapes can still be corrupted after filtering."),
        para(f"The end-to-end change was {signed(e2e.delta)}, with {number(missed_v)} missed V beats. "
             f"Record {int(concentration.record_id)} supplied {number(concentration.F)} of "
             f"{number(test.F.sum())} test F beats. The subject bootstrap "
             "describes held-out cohort variability conditional on these fitted models and noise draws. "
             "A practical extension would use recorded nonstationary artifacts such as electrode "
             "motion or muscle bursts [6]. E3 evaluates detection on the nominal recordings with the "
             "external filter fixed. This is offline beat classification on one historical database. The low S and F recall constrains clinical interpretation."),
        note("Codex assisted with implementation, visual checks and documentation. Figure preparation followed [7]. The "
             "reported values come from the archived prediction ledgers. The code guide describes "
             "the commands needed to reproduce them."),
        heading("References"),
        *[para(f"[{i}] {escape(text)} {link(url, 'Source')}.", size=8.6, leading=12.3, after=6)
          for i, (text, url) in enumerate(REFERENCES, 1)],
    ]


def guide_pages(identity, results):
    n = results["meta"]
    repo = identity["repository"]
    deps = n["dependencies"]
    return [
        [
            section("Code documentation"),
            para("Reproduce the subject-grouped MIT-BIH experiment and rebuild both PDF documents.",
                 size=11, leading=16),
            heading("Environment"),
            para(f"Verified on Linux x86-64, CPython {n['python']}, with two CPU workers. "
                 "Install Python 3.12 and uv. The transitive dependencies are recorded in "
                 "<font name='Mono'>uv.lock</font>. The workflow runs on CPU."),
            table(["Component", "Locked version / purpose"], [
                ["WFDB", f"{deps['wfdb']} · physical signal, annotations and XQRS"],
                ["NumPy / SciPy", f"{deps['numpy']} / {deps['scipy']} · arrays, DSP and feature extraction"],
                ["scikit-learn / pandas", f"{deps['scikit-learn']} / {deps['pandas']} · model pipelines and tables"],
                ["Matplotlib", f"{deps['matplotlib']} · vector figures and PNG previews"],
                ["ReportLab / PyMuPDF", f"{deps['reportlab']} / {deps['pymupdf']} · searchable vector PDFs"],
            ], [.33, .67]),
            Spacer(1, 12),
            heading("Install, download and reproduce"),
            para((f"The public source is available on GitHub. {link(repo)}." if repo else
                  "Open a terminal at the root of the supplied source project or exported source archive.")
                 + " Run the commands from that directory."),
            code((["git clone " + escape(repo), "cd " + escape(repo.rstrip("/").rsplit("/", 1)[-1])] if repo else []) + [
                "uv sync --locked",
                "uv run pytest -q",
                "uv run python -m bmsp_ecg data fetch --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg smoke --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg run --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg report --config configs/experiment.yaml --draft",
            ]),
            Spacer(1, 12),
            para("The versioned HTTPS fetch retrieves the WFDB <font name='Mono'>.dat</font>, "
                 "<font name='Mono'>.hea</font> and <font name='Mono'>.atr</font> files and verifies the "
                 "official SHA-256 list. Already valid files are reused. The original data are available "
                 f"under the Open Data Commons Attribution License v1.0. {link('https://physionet.org/content/mitdb/1.0.0/', 'PhysioNet dataset')}."),
            note("Smoke uses three training and three test records with 12 trees per branch. "
                 "The report reads the complete run. The smoke output serves only as a pipeline check."),
        ],
        [
            section("Inputs and outputs"),
            para("The experiment configuration freezes the split, label mapping, windows, filter, forest, "
                 "noise and bootstrap. The CLI also accepts <font name='Mono'>--data-dir</font> for an "
                 "existing MIT-BIH directory and <font name='Mono'>--output-dir</font> for a separate experiment root."),
            table(["Location under project root", "Contents"], [
                ["configs/experiment.yaml, configs/split.csv", "Versioned analysis and all 48 record assignments."],
                ["outputs/run/data_audit.csv, beat_counts.csv, training_beats.csv.gz",
                 "File hashes, physical lead columns, exclusions, labels, and training index."],
                ["outputs/run/reference_predictions.csv.gz", "Reference-position classifications for all clean "
                 "and noisy conditions, matched across branches by recording/sample."],
                ["outputs/run/detection_events.csv.gz", "One row per matched or missed reference plus each "
                 "extra detection. Blank sample fields denote an absent event."],
                ["outputs/run/summary.csv, per_class.csv, confusion.csv", "Protocol-level metrics, per-class "
                 "TP/FP/FN and confusion counts. The none category accounts for unmatched events."],
                ["outputs/run/noise_metrics.csv, snr_audit.csv, paired_detection.csv", "Noise grid, calibration "
                 "and paired bootstrap intervals for filtering and detection."],
                ["outputs/run/detection_by_record.csv", "Detection totals, offsets, extra detections and misses "
                 "for each record and pooled across test records."],
                ["outputs/run/*_waveform.csv, figures/", "Source samples, event tables, case metadata and "
                 "vector/PNG figures generated by the report command."],
                ["outputs/run/run_metadata.json, *_model.joblib", "Measured resources, versions, feature order "
                 "and the two fitted pipelines. Load model files only from a trusted run."],
                ["outputs/smoke/, submission/", "Smoke artifacts and the two generated documents."],
            ], [.48, .52]),
            Spacer(1, 12),
            heading("Measured full run"),
            para(f"Elapsed experiment time was <b>{n['elapsed_seconds']:.1f} s</b>. Peak process RSS "
                 f"was <b>{n['peak_rss_mib']:.0f} MiB</b>. The run generated {number(n['prediction_rows'])} "
                 f"reference-position predictions and {number(n['detection_rows'])} detection-ledger rows."),
            note("RSS is the Linux process high-water mark. Experiment timing starts after the "
                 "directory/checksum check and excludes dependency installation, downloading and PDF rendering."),
        ],
        [
            section("Verification and submission"),
            heading("Verify the evaluation ledger"),
            para("The verification utility recomputes each classification summary and detection score "
                 "from saved predictions, checks cohort separation and paired noise support, and "
                 "checks calibration and one-to-one event indices."),
            code(["uv run python scripts/verify_results.py outputs/run"]),
            Spacer(1, 12),
            heading("Build PDFs with local cover details"),
            para("Make a local <font name='Mono'>identity.yaml</font> with quoted "
                 "<font name='Mono'>name</font>, <font name='Mono'>student_id</font>, "
                 "<font name='Mono'>date</font> (YYYY-MM-DD), and <font name='Mono'>repository</font> "
                 "(public GitHub URL). The submission directory and identity file are ignored by Git."),
            code(["uv run python -m bmsp_ecg report --config configs/experiment.yaml \u005c",
                  "&nbsp;&nbsp;--identity identity.yaml"]),
            Spacer(1, 8),
            table(["PDF", "Rain Classroom field"],
                  [[REPORT, "1. report (50 points)"], [GUIDE, "2. Code (50 points)"]], [.58, .42]),
            Spacer(1, 11),
            heading("Troubleshooting"),
            para("<b>Missing or mismatched data.</b> Run <font name='Mono'>data fetch</font> again. "
                 "Existing valid files are retained, and only verified replacements are promoted. "
                 "An offline copy can be passed with <font name='Mono'>--data-dir</font>."),
            para("<b>Wrong Python.</b> Install Python 3.12 and run "
                 "<font name='Mono'>uv sync --locked --python 3.12</font>."),
            para("<b>Missing results for the report.</b> Run the full experiment first. A custom "
                 "<font name='Mono'>--output-dir</font> must be passed to both commands."),
            note("Data and method sources appear in the research report. "
                 "The study configuration and source tables are the numerical provenance for both PDFs."),
        ],
    ]


def build_reports(cfg, run, target, identity_path, draft):
    identity = load_identity(identity_path, draft)
    results = read_results(run)
    if results["meta"]["scope"] != "run":
        raise ValueError("The report requires full experiment results. Execute run first.")
    saved_config = yaml.safe_load((run / "experiment.yaml").read_text())
    if cfg != saved_config:
        raise ValueError("The active configuration differs from the saved run")
    target.mkdir(parents=True, exist_ok=True)
    figures = run / "figures"
    build_figures(run, figures)
    register_fonts()
    placements = []

    def figure(key):
        return VectorFigure(figures / f"fig_{key}.pdf", placements)

    pages = [
        cover(identity, results), acquisition(results, figure), methodology(saved_config),
        noise_results(results, figure), detection_results(results), discussion(results, figure),
        conclusion(results),
    ]
    build_pdf(target / REPORT, pages, TITLE + " " + SUBTITLE, "Course project report",
              identity["name"], draft, placements)
    build_pdf(target / GUIDE, guide_pages(identity, results), "BMSP Code Documentation",
              "Code documentation", identity["name"], draft, [])
    print(f"Wrote {target / REPORT} and {target / GUIDE}")
