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
from .layout import (COVER_LIGHT, INK, TEAL, CoverTrace, Spacer, VectorFigure,
                     build_pdf, code, heading, note, para, register_fonts, section, table)
from .signal import BRANCH_LABELS, CLASSES, filter_sos
from .stress import checked_stress

TITLE = "How Much Does Filtering Help?"
SUBTITLE = "Noise-Robust ECG Beat Classification on Unseen Subjects"
REPORT = "BMSP_Report.pdf"
GUIDE = "BMSP_Code_Documentation.pdf"
NO_BANDPASS, WITH_BANDPASS = (BRANCH_LABELS[name] for name in ("raw", "filtered"))


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
    identity["name_native"] = str(value.get("name_native") or "").strip()
    return identity


def student_name(identity):
    full_name = " · ".join(part for part in (identity["name"], identity["name_native"]) if part)
    return re.sub(
        r"[\u3400-\u9fff]+", lambda match: f'<font name="StudentCJK">{match.group()}</font>',
        escape(full_name),
    )


def read_results(run):
    stress = checked_stress(run)
    return {
        "meta": json.loads((run / "run_metadata.json").read_text()),
        "case": json.loads((run / "failure_meta.json").read_text()),
        "case_events": pd.read_csv(run / "failure_events.csv"),
        "scores": pd.read_csv(run / "summary.csv").set_index("protocol"),
        "noise": pd.read_csv(run / "noise_metrics.csv", dtype={"snr_db": str}, keep_default_na=False),
        "counts": pd.read_csv(run / "beat_counts.csv"),
        "audit": pd.read_csv(run / "data_audit.csv", dtype={"record_id": str}),
        "class": pd.read_csv(run / "per_class.csv"),
        "matrix": pd.read_csv(run / "confusion.csv", keep_default_na=False),
        "detection": pd.read_csv(run / "detection_by_record.csv", dtype={"record_id": str}).set_index("record_id"),
        "paired": pd.read_csv(run / "paired_detection.csv").set_index("comparison"),
        "snr": pd.read_csv(run / "snr_audit.csv"),
        "sensitivity": pd.read_csv(run / "sensitivity_record_213.csv").set_index("scope"),
        "minority": pd.read_csv(run / "minority_errors.csv").set_index(["measure", "true", "outcome"]).sort_index(),
        "stress": (pd.read_csv(stress / "metrics.csv").set_index("branch"),
                   pd.read_csv(stress / "paired_delta.csv").iloc[0]) if stress else None,
    }


def cover(identity, results, trace):
    n = results["meta"]
    scores = results["scores"]
    raw, filtered = (scores.loc[key].macro_f1 for key in ("reference_raw", "reference_filtered"))
    noise = results["noise"]
    crossings = ((noise.lo <= 0) & (noise.hi >= 0)).sum()
    drift = noise.query("noise == 'drift' and snr_db == '0'").iloc[0]
    replay = (f" A separate 0 dB drift replay with XQRS changed end-to-end macro-F1 by "
              f"{signed(results['stress'][1].delta)}." if results["stress"] else "")
    repo = identity["repository"]
    return [
        Spacer(1, 12),
        para("Course project report  /  Fall 2026", size=10, leading=14, face="BodyBold",
             color=COVER_LIGHT, after=15),
        para("How Much Does<br/>Filtering Help?", size=40, leading=43, face="SerifBold",
             color="#FFFFFF", after=14),
        para("Noise-robust ECG beat classification on unseen subjects",
             size=16.2, leading=21, face="Serif", color="#E3F2EE", after=15),
        trace,
        Spacer(1, 61),
        para(student_name(identity) or "Student name pending", size=15, leading=20, face="BodyBold",
             color=INK, after=4),
        para(f"Student ID {escape(identity['student_id'] or 'pending')} &nbsp;·&nbsp; "
             f"{escape(identity['date'] or date.today().isoformat())}", after=3),
        para("Huaqiao University &nbsp;·&nbsp; Instructor Dr. DUAN Fang", after=4),
        para(f"Source code &nbsp; {link(repo) if repo else 'Public repository pending'}", after=23),
        heading("Abstract"),
        para(f"Two random forests classified MLII beats from {number(n['test_subjects'])} held-out "
             f"MIT-BIH subjects. The noise comparison used reference beat positions. Without added noise, "
             f"macro-F1 was {f1(raw)} for <i>{NO_BANDPASS}</i> and {f1(filtered)} for "
             f"<i>{WITH_BANDPASS}</i>. The paired 95% intervals included zero in {crossings} of {len(noise)} "
             f"conditions, including 0 dB baseline drift (difference {signed(drift.delta)}, {ci(drift)}). "
             f"The primary XQRS experiment used recordings without added noise and yielded an end-to-end "
             f"macro-F1 of {f1(scores.loc['end_to_end'].macro_f1)} with the external band-pass.{replay}"),
    ]


def acquisition(results, figure):
    counts = results["counts"].groupby("partition").sum(numeric_only=True)
    train, test = counts.loc["train"], counts.loc["test"]
    n = results["meta"]
    missing = results["audit"].query("status == 'missing_mlii'").record_id.tolist()
    columns = results["audit"].query("status == 'included' and mlii_column_zero_based == 1").record_id.tolist()
    lead_note = ("Records without MLII were " + ", ".join(missing) + ".") if missing else (
        "MLII was available in all 44 eligible records.")
    return [
        section("Data and study design"),
        para("MIT-BIH v1.0.0 contains 48 half-hour ambulatory ECG recordings from 47 people. Each "
             "recording has two channels sampled at 360 Hz, with 11-bit resolution over a 10 mV range. "
             "Cardiologists independently annotated the beats and reconciled their readings [1, 2]."),
        para("WFDB returned amplitudes in mV. Lead selection used the header name MLII, since record "
             f"{', '.join(columns)} stores it in the second channel. {lead_note} The data audit lists "
             f"the selected lead and annotation counts and verifies {n['official_hashes']} published file hashes."),
        heading("Training and test subjects"),
        para("The split starts from DS1/DS2 [3]. Records 102, 104, 107 and 217 were excluded because of pacing. "
             "Record 202 moved to training to keep it with record 201, which came from the same person [1]. "
             "The resulting training and test sets share no subjects."),
        table(["Partition", "Records", "Subjects", "N", "S", "V", "F", "N/S/V/F total"],
              [["Training", n["training_records"], n["training_subjects"], *[number(train[x]) for x in CLASSES],
                number(sum(train[x] for x in CLASSES))],
               ["Held-out test", n["test_subjects"], n["test_subjects"], *[number(test[x]) for x in CLASSES],
                number(sum(test[x] for x in CLASSES))]],
              [.16, .115, .125, .115, .095, .10, .09, .20], numeric=range(1, 8)),
        Spacer(1, 8),
        note(f"Table 1. Four-class beat counts after windowing. Training/test exclusions were "
             f"{number(train.Q)}/{number(test.Q)} class-Q beats, {number(train.unmapped_qrs)}/"
             f"{number(test.unmapped_qrs)} unmapped QRS beats and {number(train.qrs_without_window)}/"
             f"{number(test.qrs_without_window)} beats at record edges. Non-QRS annotations numbered "
             f"{number(train.non_qrs_events)}/{number(test.non_qrs_events)}."),
        heading("MLII with and without the external band-pass"),
        figure("qc"),
        note("Figure 1. Three seconds from training record 114 on a shared mV scale. "
             "Dashed lines mark reference beat times."),
    ]


def methodology(spec):
    fs = spec["dataset"]["fs_hz"]
    frequencies = [0.33, 0.5, 10, 40, 75]
    _, response = sosfreqz(filter_sos(fs, spec["filter"]), worN=frequencies, fs=fs)
    gains = np.round(40 * np.log10(np.abs(response)), 1) + 0.0
    return [
        section("Methodology"),
        heading("Band-pass filter and beat windows"),
        para(f"The <i>{NO_BANDPASS}</i> branch takes the recorded MLII signal as input. "
             f"<i>{WITH_BANDPASS}</i> applies a 0.5–40 Hz Butterworth band-pass to the complete record. "
             "A fourth-order prototype gives four biquads for the "
             "single pass. Forward–backward application cancels phase and squares the magnitude response [4]. "
             f"The effective gains at 0.33, 0.5, 10, 40 and 75 Hz are "
             f"{', '.join(f'{g:.1f}' for g in gains)} dB, respectively."),
        para("Both branches use the same 28 timing and amplitude features. The beat window contains "
             "216 samples from −250 ms inclusive to +350 ms exclusive, relative to the beat center. "
             "For morphology, both branches subtract the local median from −200 to −120 ms."),
        table(["Feature group", "Definition and units"], [
            ["RR (4)", "Previous and next intervals between QRS positions (s), and each interval divided "
             "by its centered 11-interval median. "
             "Timing uses recognized QRS positions, including those too close to a record edge to classify."],
            ["QRS (8)", "Center amplitude (mV). Within −80 to +100 ms, calculate the minimum, maximum, "
             "peak-to-peak and RMS amplitude (mV), absolute area (mV·s), time at or above half-peak (s) "
             "and maximum absolute slope (mV/s)."],
            ["Whole beat (2)", "RMS and peak-to-peak amplitude (mV)."],
            ["Binned shape (14)", "Mean amplitude (mV) in consecutive 30 ms bins from −180 to +240 ms."],
        ], [.23, .77]),
        Spacer(1, 8),
        note("Table 2. Time bounds are rounded to the nearest sample. The RR median repeats the nearest "
             "interval at a record edge. A beat without an adjacent QRS has a missing RR feature."),
        heading("Labels and model fitting"),
        para("The class mapping follows de Chazal et al. [3]. N includes N/L/R/e/j, S includes A/a/J/S, "
             "V includes V/E, and F includes F. The Q group contains /, f and Q. The audit tracks "
             "B, n and r as unmapped. Training uses reference QRS centers."),
        para("Each forest has 300 trees, class weights balanced within each bootstrap sample, square-root "
             "feature sampling, at least two samples per leaf and seed 877. Both use the same training beats "
             "without added noise. Imputation medians come from training only. The hyperparameters were fixed."),
        para("Macro-F1 averages the four class F1 scores with equal weights. The analysis is offline because "
             "the band-pass and following RR intervals use future samples.", after=0),
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
        section("Results at reference beats"),
        para("To compare preprocessing alone, both classifiers used the reference beat centers. Each test "
             "record received either zero-mean white Gaussian noise or a 0.33 Hz sinusoid with a seeded "
             "phase. The signal-to-noise ratios (SNRs) were 24, 12, 6 and 0 dB. The noise scale was "
             "√(P<sub>signal</sub> / (P<sub>noise</sub> × 10<super>SNR/10</super>)), "
             "where signal power came from the demeaned original. Both branches received the same disturbed "
             f"waveform. The achieved SNR was within {max(discrepancy, 1e-9):.0e} dB of its target."),
        para(f"On the original recordings, predicting N for every beat gave {pct(base.accuracy)} accuracy "
             f"and {f1(base.macro_f1)} macro-F1. <i>{NO_BANDPASS}</i> scored {f1(test.macro_f1)} "
             f"and <i>{WITH_BANDPASS}</i> scored {f1(filt.macro_f1)}."),
        figure("robustness"),
        note(f"Figure 2. Each condition uses the same {number(test.support)} reference-position beats "
             f"from {number(results['meta']['test_subjects'])} test subjects. Δ is with minus without "
             "the external band-pass. Bars span "
             "the 2.5th to 97.5th percentiles of 2,000 paired subject resamples with both forests fixed. "
             "A class with a zero F1 denominator scores zero. SNR is relative to the recorded ECG."),
        table(["Selected condition", NO_BANDPASS, WITH_BANDPASS, "Δ Macro-F1", "95% paired interval"], [
            [name, f1(row.raw_f1), f1(row.filtered_f1), signed(row.delta), ci(row)]
            for name, row in zip(["No added noise", "White noise · 0 dB", "Baseline drift · 0 dB"], extrema)
        ], [.225, .19, .19, .13, .265], numeric=range(1, 5)),
        Spacer(1, 7),
        note("Table 3. Estimates without added noise and at the strongest added noise level."),
        para(f"With the external band-pass, macro-F1 stayed between {f1(drift_levels.min())} and "
             f"{f1(drift_levels.max())} across the four drift levels. "
             f"The paired intervals included zero in {crossing} of {len(noise)} conditions. "
             f"On the original recordings, fusion F1 changed from {f1(raw_f)} to {f1(filtered_f)} with the band-pass, "
             "the largest class-level contribution to the drop in macro-F1."),
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
        section("XQRS results without added noise"),
        para(f"The primary detector comparison used the recordings without added noise and the "
             f"<i>{WITH_BANDPASS}</i> branch. XQRS "
             "from WFDB 4.3.1 ran with its default settings, including signal-based initialization and a "
             "5–20 Hz internal band-pass [5]. In this branch, detected positions set the RR intervals "
             "and waveform windows. The reference and detection branches used the same window boundaries."),
        para("Pairs within ±150 ms (54 samples) were matched one to one. The matcher maximized the number "
             "of pairs first and minimized total absolute timing error second. A missed labeled beat adds "
             "a false negative and an extra detection adds a false positive for the predicted class. "
             "A wrong class adds one of each. Q and unmapped beats can match for the detection score."),
        para(f"XQRS matched {number(d.tp)} reference beats, missed {number(d.fn)} and reported {number(d.fp)} "
             f"extra detections. Precision was {pct(d.precision)}, recall was {pct(d.recall)} and F1 was "
             f"{f1(d.f1)}. The detection tally includes {number(d.matched_excluded)} matches to excluded classes."),
        table(["Position source", "Evaluation support", "N/S/V/F refs", "Macro-F1"], [
            [title, subset, number(scores.loc[key].support), f1(scores.loc[key].macro_f1)]
            for key, title, subset in groupings
        ], [.23, .46, .16, .15], numeric=(2, 3)),
        Spacer(1, 7),
        note("Table 4. The middle rows use the same beats to compare reference and detected centers. "
             "The final row includes misses and extra detections."),
        para(f"Matched offsets had a median absolute error of {d.median_abs_ms:.1f} ms and a 95th "
             f"percentile of {d.p95_abs_ms:.1f} ms. Using detected positions for those same beats "
             f"changed macro-F1 by {signed(cond.delta)} (95% interval {ci(cond)}). Including errors "
             f"on all beats changed it by {signed(e2e.delta)} (95% interval {ci(e2e)})."),
        table(["Class", "Test beats", "Always-N recall", f"{NO_BANDPASS} recall", f"{WITH_BANDPASS} recall",
               f"{WITH_BANDPASS} precision", f"{WITH_BANDPASS} F1", "End-to-end F1"],
              [[label, number(base.loc[label].support), pct(constant.loc[label].recall),
                pct(raw.loc[label].recall), pct(base.loc[label].recall),
                pct(base.loc[label].precision), f1(base.loc[label].f1), f1(end.loc[label].f1)]
               for label in CLASSES], [.09, .115, .145, .125, .125, .14, .12, .14], numeric=range(1, 8)),
        Spacer(1, 7),
        note("Table 5. Class results without added noise. Both band-pass columns and the always-N "
             "baseline use reference positions. End-to-end F1 uses the external band-pass and XQRS. "
             "N includes normal and similar beats, S is supraventricular ectopic, V is ventricular ectopic "
             "and F is fusion."),
    ]


def stress_results(results):
    metrics, delta = results["stress"]
    drift = results["noise"].query("noise == 'drift' and snr_db == '0'").iloc[0]
    raw, filtered = (metrics.loc[branch] for branch in ("raw", "filtered"))
    same_qrs = raw[["matched", "missed", "extra"]].equals(filtered[["matched", "missed", "extra"]])
    detection_note = (f"Both branches matched {number(raw.matched)} QRS beats, missed "
                      f"{number(raw.missed)} and added {number(raw.extra)} detections."
                      if same_qrs else
                      f"QRS F1 was {f1(raw.qrs_f1)} without and {f1(filtered.qrs_f1)} with the external band-pass.")
    return [
        section("XQRS with 0 dB baseline drift"),
        para("This replay adds 0.33 Hz baseline drift at 0 dB to the same held-out recordings. Each "
             "record receives the same injected waveform in both branches. The two saved forests are "
             "applied without refitting. XQRS runs separately on the input of each branch, followed by "
             "the same beat-window and classification steps."),
        para("The matching tolerance and counting rules are the same as for recordings without "
             "added noise. For each class, end-to-end recall is the fraction of its reference beats "
             "that are both detected and classified correctly."),
        table(["Branch", "QRS F1", "End-to-end macro-F1", "S recall", "V recall"], [
            [BRANCH_LABELS[branch], f1(row.qrs_f1), f1(row.macro_f1), pct(row.s_recall), pct(row.v_recall)]
            for branch, row in metrics.iterrows()
        ], [.33, .12, .23, .16, .16], numeric=range(1, 5)),
        Spacer(1, 8),
        note(f"Table 6. Additional experiment on {number(raw.support)} four-class reference beats. "
             "Both detector branches still use XQRS's internal preprocessing."),
        heading("Does the estimated gain persist?"),
        para(f"At reference positions, the 0 dB drift estimate was {signed(drift.delta)} macro-F1 "
             f"(95% paired interval {ci(drift)}). With XQRS and with detection errors counted, the "
             f"estimate was {signed(delta.delta)} ({ci(delta)}). The positive point estimate persisted "
             "after automatic localization. Both intervals included zero."),
        para(detection_note + " This comparison covers a sinusoidal drift at one noise level."),
        note(f"The paired interval uses {number(delta.bootstrap_repetitions)} bootstrap resamples of "
             f"the {number(delta.subjects)} held-out subjects."),
    ]


def classification_discussion(results, first_table):
    matrix = results["matrix"].query("protocol == 'reference_filtered'").pivot(
        index="true", columns="pred", values="count",
    )
    full, reduced = (results["sensitivity"].loc[scope] for scope in ("all", "without_213"))
    minority = results["minority"].loc["reference_filtered_classification", "S"]
    correct, as_n = (minority.loc[label] for label in ("S", "N"))
    return [
        section("Discussion"),
        heading("Post hoc sensitivity to record 213"),
        table(["Test records", "Beats", f"{NO_BANDPASS}<br/>Macro-F1", f"{WITH_BANDPASS}<br/>Macro-F1"], [
            [scope, number(row.beats), f1(row.raw_f1), f1(row.filtered_f1)]
            for scope, row in (("All test records", full), ("Excluding record 213", reduced))
        ], [.28, .14, .29, .29], numeric=range(1, 4)),
        Spacer(1, 8),
        note(f"Table {first_table}. Recomputed from saved predictions at reference beat positions "
             "without added noise. The full test set remains the primary result."),
        para(f"Record 213 contributed {number(full.fusion_beats - reduced.fusion_beats)} of the "
             f"{number(full.fusion_beats)} fusion beats in the full test set. The with-minus-without "
             f"difference was {signed(full.delta)} on the full set and {signed(reduced.delta)} "
             "after removing record 213. The two branches were nearly tied on the reduced set. "
             "The observed difference without added noise is sensitive to the test cohort's record composition."),
        heading("Classification errors at reference beats"),
        table(["Reference / prediction", *CLASSES], [
            [label, *[number(matrix.loc[label, pred]) for pred in CLASSES]]
            for label in CLASSES
        ], [.36, .16, .16, .16, .16], numeric=range(1, 5)),
        Spacer(1, 8),
        note(f"Table {first_table + 1}. {WITH_BANDPASS} at reference positions without added noise. "
             "Rows are annotated classes and columns are predictions."),
        para(f"Of the {number(correct.support)} S beats, the classifier labeled {number(correct['count'])} "
             f"as S and {number(as_n['count'])} as N. The S-to-N error is the main classification problem "
             "for S in this test set. A supraventricular ectopic beat can resemble N in QRS shape [3], "
             "so the next feature comparison should test what helps distinguish S from N."),
    ]


def detection_discussion(results, figure, table_number):
    case = results["case"]
    label = case["focus_label"]
    t0, t1, centre = (case[key] / case["fs"] for key in ("start", "stop", "focus_sample"))
    condition = ("A missed reference beat" if case["focus_status"] == "miss" else
                 f"A reference beat predicted as {case['focus_prediction']}")
    confusion = results["matrix"].query("protocol == 'end_to_end'").pivot(index="true", columns="pred", values="count")
    errors = results["minority"].loc["xqrs_detection_without_added_noise"]
    v_miss, n_miss = (errors.loc[(label, "miss")] for label in ("V", "N"))
    qrs = results["detection"].loc["ALL"]
    focus = case["focus_sample"]
    references = results["case_events"].ref_sample
    detections = results["case_events"].det_sample
    left_ref, right_ref = references[references < focus].max(), references[references > focus].min()
    left_det, right_det = detections[detections < focus].max(), detections[detections > focus].min()
    intervals = np.array([focus - left_ref, right_ref - focus, right_det - left_det]) / case["fs"]
    if case["focus_status"] == "miss" and np.isfinite(intervals).all():
        prev_rr, next_rr, det_rr = intervals
        case_insight = (
            f"The reference RR intervals on either side of the missed beat were {prev_rr:.3f} s and "
            f"{next_rr:.3f} s. XQRS measured a single {det_rr:.3f} s interval between the flanking "
            "detections. In addition to losing the ventricular beat, the miss changes the RR feature "
            "available for each neighbor. The trace cannot establish what caused the detector to miss it."
        )
    else:
        case_insight = ("A missed center changes the RR interval used for neighboring detections. "
                        "Both the recovered timing and the choice of waveform center enter the classifier.")
    return [
        section("Discussion of missed beats"),
        figure("failure"),
        note(f"Figure 3. {condition} ({label}) at {centre:.3f} s in test record "
             f"{case['record_id']} (sample {focus:,}). The excerpt spans [{t0:.3f}, {t1:.3f}) s. "
             "Dashed lines mark references, triangles mark detections and shading marks the match window. "
             "The rule selects the earliest eligible V miss in the lowest-numbered test record, "
             "at least two seconds from a record edge."),
        para(case_insight),
        para(f"XQRS reached a pooled detection F1 of {f1(qrs.f1)} without added noise. "
             f"It missed {number(v_miss['count'])}/{number(v_miss.support)} V beats "
             f"({pct(v_miss.rate)}) and {number(n_miss['count'])}/{number(n_miss.support)} N beats "
             f"({100 * n_miss.rate:.3f}%). The missed-beat rate was appreciably higher for V."),
        table(["Reference / prediction", "N", "S", "V", "F", "No detection"],
              [[label if label != "none" else "Extra detection",
                *[number(confusion.loc[label, pred]) for pred in (*CLASSES, "none")]]
               for label in (*CLASSES, "none")], [.25, .145, .12, .12, .12, .245], numeric=range(1, 6)),
        Spacer(1, 7),
        note(f"Table {table_number}. End-to-end errors without added noise. Misses occupy the last column "
             "and extra detections the last row. Matches to excluded classes count toward detection only."),
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
    noise = results["noise"]
    drift = noise.query("noise == 'drift' and snr_db == '0'").iloc[0]
    crossings = ((noise.lo <= 0) & (noise.hi >= 0)).sum()
    e2e = results["paired"].loc["reference_to_end_to_end"]
    replay = results["stress"][1] if results["stress"] else None
    replay_text = (f" The separate 0 dB drift replay gave an end-to-end difference of "
                   f"{signed(replay.delta)} ({ci(replay)})." if replay is not None else "")
    return [
        section("Conclusion"),
        para(f"With reference positions fixed, the 95% paired intervals included zero in "
             f"{crossings} of {len(noise)} conditions. At 0 dB baseline drift the estimated gain "
             f"from the external band-pass was {signed(drift.delta)} macro-F1 (95% interval {ci(drift)}). "
             "Both branches subtract a local baseline for morphology. White noise still overlaps the passband."),
        para(f"In the primary end-to-end experiment without added noise, replacing reference positions "
             f"with XQRS and counting misses and extra detections changed macro-F1 by {signed(e2e.delta)}."
             f"{replay_text}"),
        heading("Limits of the comparison"),
        para("The no-added-noise comparison was sensitive to one record with most of the fusion beats. "
             "The paired bootstrap captures this cohort's variation for the fitted models and fixed "
             "noise draws. The noise was synthetic. A test with recorded electrode motion or muscle "
             "artifact would address a different use case [6]."),
        note("Codex assisted with implementation, editing and figure preparation. The visualization "
             "workflow drew on Scientific Agent Skills [7]."),
        heading("References"),
        *[para(f"[{i}] {escape(text)} {link(url, url.removeprefix('https://') if 'doi.org/' in url else 'Source')}.",
               size=8.6, leading=12.3, after=6)
          for i, (text, url) in enumerate(REFERENCES, 1)],
    ]


def guide_pages(identity, results):
    n = results["meta"]
    repo = identity["repository"]
    deps = n["dependencies"]
    name = student_name(identity)
    owner_line = [
        note(f"{name} &nbsp;·&nbsp; Student ID {escape(identity['student_id'])} "
             f"&nbsp;·&nbsp; {escape(identity['date'])}")
    ] if name else []
    return [
        [
            section("Code documentation"),
            para("MIT-BIH beat classification and report generation", size=11, leading=16, after=5),
            *owner_line,
            heading("Environment"),
            para(f"Use Linux or WSL with Python 3.12 and uv. The reference run used Linux x86-64, "
                 f"CPython {n['python']} and two CPU workers. <font name='Mono'>uv.lock</font> pins "
                 "the full dependency tree."),
            table(["Component", "Version and role"], [
                ["WFDB", f"{deps['wfdb']} · physical signal, annotations and XQRS"],
                ["NumPy / SciPy", f"{deps['numpy']} / {deps['scipy']} · arrays, DSP and feature extraction"],
                ["scikit-learn / pandas", f"{deps['scikit-learn']} / {deps['pandas']} · model pipelines and tables"],
                ["Matplotlib", f"{deps['matplotlib']} · vector figures and PNG previews"],
                ["ReportLab / PyMuPDF", f"{deps['reportlab']} / {deps['pymupdf']} · searchable vector PDFs"],
            ], [.33, .67]),
            Spacer(1, 12),
            heading("Run the experiment"),
            para(f"Clone the public repository and run the commands below. {link(repo)}" if repo else
                 "Run the commands from the project root."),
            code((["git clone " + escape(repo), "cd " + escape(repo.rstrip("/").rsplit("/", 1)[-1])] if repo else []) + [
                "uv sync --locked",
                "uv run pytest -q",
                "uv run python -m bmsp_ecg data fetch --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg smoke --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg run --config configs/experiment.yaml",
                "uv run python -m bmsp_ecg stress --config configs/experiment.yaml",
                "uv run python scripts/verify_results.py outputs/run",
                "uv run python -m bmsp_ecg report --config configs/experiment.yaml --draft",
            ]),
            Spacer(1, 12),
            para("The fetch command downloads the WFDB <font name='Mono'>.dat</font>, "
                 "<font name='Mono'>.hea</font> and <font name='Mono'>.atr</font> files. It checks each "
                 "against the published SHA-256 list and reuses files that already match. MIT-BIH is available "
                 f"under the Open Data Commons Attribution License v1.0. {link('https://physionet.org/content/mitdb/1.0.0/', 'PhysioNet dataset')}."),
            note("Smoke writes to outputs/smoke/. The full experiment writes to outputs/run/. "
                 "Stress reuses the saved models and writes to outputs/run/stress/. Both PDFs go to submission/."),
        ],
        [
            section("Inputs and outputs"),
            para("Use <font name='Mono'>--data-dir</font> to reuse a local MIT-BIH copy. "
                 "<font name='Mono'>--output-dir</font> changes the results directory."),
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
                ["outputs/run/sensitivity_record_213.csv, minority_errors.csv",
                 "Post hoc clean reference-position comparison and minority-class counts."],
                ["outputs/run/stress/", "0 dB drift detector ledger, paired waveform audit and metrics."],
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
            para(f"The run took {n['elapsed_seconds']:.1f} s and peaked at {n['peak_rss_mib']:.0f} MiB RSS. "
                 f"It wrote {number(n['prediction_rows'])} reference-position predictions and "
                 f"{number(n['detection_rows'])} detection-ledger rows."),
            note("Timing covers the experiment after the data check. RSS is the Linux process high-water mark."),
        ],
        [
            section("Verification and submission"),
            heading("Verify the evaluation ledger"),
            para("The verifier recalculates the main results, the record 213 sensitivity table, the "
                 "minority-class counts and the drift replay metrics from saved predictions and matches."),
            code(["uv run python scripts/verify_results.py outputs/run"]),
            para("To rebuild the two post hoc tables from a saved full run, use the analyze command. "
                 "It reads prediction files only."),
            code(["uv run python -m bmsp_ecg analyze --config configs/experiment.yaml"]),
            Spacer(1, 12),
            heading("Build PDFs with local cover details"),
            para("Save a local <font name='Mono'>identity.yaml</font> with quoted "
                 "<font name='Mono'>name</font>, <font name='Mono'>student_id</font>, "
                 "<font name='Mono'>date</font> (YYYY-MM-DD), and <font name='Mono'>repository</font> "
                 "(public GitHub URL). Use <font name='Mono'>name_native</font> for a second spelling if "
                 "needed. Git ignores the identity file and the submission directory."),
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
    names = f"{identity['name']} {identity['name_native']}"
    register_fonts(native_name=bool(re.search(r"[\u3400-\u9fff]", names)))
    placements = []

    def figure(key):
        return VectorFigure(figures / f"fig_{key}.pdf", placements)

    qc = pd.read_csv(run / "qc_waveform.csv")
    qc_meta = json.loads((run / "qc_meta.json").read_text())
    trace = CoverTrace(qc.time_s.to_numpy(), qc.raw_mV.to_numpy(), qc_meta["record_id"])
    pages = [
        cover(identity, results, trace), acquisition(results, figure), methodology(saved_config),
        noise_results(results, figure), detection_results(results),
    ]
    if results["stress"]:
        pages.append(stress_results(results))
    first_table = 7 if results["stress"] else 6
    pages.extend([
        classification_discussion(results, first_table),
        detection_discussion(results, figure, first_table + 2),
        conclusion(results),
    ])
    build_pdf(target / REPORT, pages, TITLE + " " + SUBTITLE, "Course project report",
              identity["name"], draft, placements, cover_band=True)
    build_pdf(target / GUIDE, guide_pages(identity, results), "BMSP Code Documentation",
              "Code documentation", identity["name"], draft, [])
    print(f"Wrote {target / REPORT} and {target / GUIDE}")
