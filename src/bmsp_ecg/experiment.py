"""Train two locked branches, then record all evaluations at beat resolution."""

import json
import platform
import resource
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import wfdb.processing
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .data import load_record, sha256
from .matching import match_qrs
from .metrics import summarize
from .signal import CLASSES, FEATURE_NAMES, bandpass, feature_matrix, inject, symbol_classes, unit_noise, window_valid

# Contains V and rare F training examples, and covers the reversed MLII column.
SMOKE = {"101", "114", "208", "100", "200", "213"}


@dataclass
class Reference:
    centres: np.ndarray
    labels: np.ndarray
    symbols: np.ndarray


def references(record, cfg):
    labels = symbol_classes(record.symbols, cfg["labels"])
    is_beat = labels != "event"
    return Reference(record.samples[is_beat], labels[is_beat], record.symbols[is_beat])


def classification_frame(record, ref, rows, labels, noise, snr, branch):
    return pd.DataFrame(dict(
        record_id=record.split.record_id, subject_group=record.split.subject_group,
        noise=noise, snr_db=snr, branch=branch, sample=ref.centres[rows],
        symbol=ref.symbols[rows], true=ref.labels[rows], pred=labels,
    ))


def reference_predictions(record, ref, cfg, branches, waveform, noise, snr):
    if noise == "clean":
        assert snr == "none"
    for branch, model in branches.items():
        source = bandpass(waveform, record.fs, cfg["filter"]) if branch == "filtered" else waveform
        features, valid = feature_matrix(source, ref.centres, record.fs, cfg["window_s"])
        keep = np.isin(ref.labels[valid], CLASSES)
        retained = np.flatnonzero(valid)[keep]
        yield classification_frame(record, ref, retained, model.predict(features[keep]), noise, snr, branch)
        if branch == "raw" and noise == "clean":
            all_n = np.repeat("N", len(retained))
            yield classification_frame(record, ref, retained, all_n, noise, snr, "all_n")


def detector_events(record, ref, filtered, model, cfg):
    fs = record.fs
    detections = wfdb.processing.xqrs_detect(
        sig=filtered, fs=fs, conf=None, learn=cfg["detection"]["learn"], verbose=False,
    )
    features, valid_d = feature_matrix(filtered, detections, fs, cfg["window_s"])
    pred = model.predict(features) if len(features) else np.array([], dtype="U1")
    valid_r = window_valid(ref.centres, len(filtered), fs, cfg["window_s"])
    true_samples = ref.centres[valid_r]
    det_samples = detections[valid_d]
    labels = ref.labels[valid_r]
    symbols = ref.symbols[valid_r]
    tolerance = round(cfg["detection"]["tolerance_ms"] * fs / 1000)
    pairs = match_qrs(true_samples, det_samples, tolerance)
    ref2det = dict(pairs)
    matched_det = {j for _, j in pairs}
    rows = []
    for i, sample in enumerate(true_samples):
        j = ref2det.get(i)
        common = dict(ref_sample=sample, symbol=symbols[i], true=labels[i])
        if j is None:
            rows.append(dict(**common, status="miss", det_sample=None, pred="none", shift_samples=None))
        else:
            rows.append(dict(**common, status="matched", det_sample=det_samples[j], pred=pred[j],
                             shift_samples=det_samples[j] - sample))
    rows.extend(dict(ref_sample=None, symbol="", true="none", status="false_positive",
                     det_sample=sample, pred=pred[j], shift_samples=None)
                for j, sample in enumerate(det_samples) if j not in matched_det)
    return pd.DataFrame(rows).assign(
        record_id=record.split.record_id, subject_group=record.split.subject_group,
    ), int((~valid_d).sum())


def save_segment(target, name, record, filtered, ref, events=None, case=None):
    focus = int(case.ref_sample if case is not None else ref.centres[ref.centres >= 4 * record.fs][0])
    half_width = round(1.5 * record.fs)
    a, b = focus - half_width, focus + half_width
    samples = np.arange(a, b)
    pd.DataFrame(dict(sample=samples, time_s=samples / record.fs, raw_mV=record.signal[a:b],
                      filtered_mV=filtered[a:b])).to_csv(target / f"{name}_waveform.csv", index=False)
    mask = (ref.centres >= a) & (ref.centres < b)
    pd.DataFrame(dict(sample=ref.centres[mask], symbol=ref.symbols[mask], label=ref.labels[mask])).to_csv(
        target / f"{name}_reference.csv", index=False,
    )
    if events is not None:
        record_events = events[events.record_id == record.split.record_id]
        included = record_events[record_events.ref_sample.between(a, b - 1) |
                                 record_events.det_sample.between(a, b - 1)]
        included.to_csv(target / f"{name}_events.csv", index=False)
    (target / f"{name}_meta.json").write_text(json.dumps(dict(
        record_id=record.split.record_id, start=a, stop=b, fs=record.fs, focus_sample=focus, unit=record.unit,
        focus_label=case.true if case is not None else "",
        focus_status=case.status if case is not None else "",
        focus_prediction=case.pred if case is not None else "",
    ), indent=2) + "\n")


def select_case(events, records):
    for label, status in (("V", "miss"), ("S", "miss"), ("N", "miss"), ("V", "matched")):
        mask = (events.true == label) & (events.status == status)
        if status == "matched":
            mask &= events.true != events.pred
        for case in events.loc[mask].sort_values(["record_id", "ref_sample"]).itertuples(index=False):
            record = records[case.record_id]
            if 2 * record.fs <= case.ref_sample <= len(record.signal) - 2 * record.fs:
                return case
    raise ValueError("No eligible error waveform in this scope")


def run_experiment(cfg, root, rows, checksums, target, scope):
    started = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{scope}-", dir=target.parent) as temporary:
        staged = Path(temporary)
        _run(cfg, root, rows, checksums, staged, scope, started)
        target.mkdir(exist_ok=True)
        for file in staged.iterdir():
            file.replace(target / file.name)
    elapsed = time.perf_counter() - started
    print(f"{scope} completed in {elapsed:.2f} s; artifacts: {target}", flush=True)


def _run(cfg, root, rows, checksums, target, scope, started):
    audit, records, counts, training_index = [], {}, [], []
    inputs = {"raw": [], "filtered": []}
    for row in rows:
        if scope == "smoke" and row.record_id not in SMOKE:
            continue
        record, audited = load_record(root, row, cfg["dataset"], checksums)
        audit.append(audited)
        if record is None:
            continue
        records[row.record_id] = record
        ref = references(record, cfg)
        raw_features, valid = feature_matrix(record.signal, ref.centres, record.fs, cfg["window_s"])
        keep = np.isin(ref.labels[valid], CLASSES)
        counts.append(dict(record_id=row.record_id, partition=row.partition,
                           qrs_with_window=int(valid.sum()), qrs_without_window=int((~valid).sum()),
                           **{label: int((ref.labels[valid] == label).sum()) for label in
                              (*CLASSES, "Q", "unmapped_qrs")},
                           non_qrs_events=len(record.samples) - len(ref.centres)))
        if row.partition != "train":
            continue
        filtered = bandpass(record.signal, record.fs, cfg["filter"])
        filtered_features, same = feature_matrix(filtered, ref.centres, record.fs, cfg["window_s"])
        if not np.array_equal(same, valid):
            raise RuntimeError("Training support differs across branches")
        inputs["raw"].append(raw_features[keep])
        inputs["filtered"].append(filtered_features[keep])
        training_index.append(pd.DataFrame(dict(
            record_id=row.record_id, subject_group=row.subject_group, sample=ref.centres[valid][keep],
            true=ref.labels[valid][keep],
        )))
        if row.record_id == cfg["detection"]["qc_record"]:
            save_segment(target, "qc", record, filtered, ref)
    pd.DataFrame(audit).to_csv(target / "data_audit.csv", index=False)
    pd.DataFrame(counts).to_csv(target / "beat_counts.csv", index=False)
    index = pd.concat(training_index, ignore_index=True)
    index.to_csv(target / "training_beats.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    train_groups = set(index.subject_group)
    test_groups = {r.split.subject_group for r in records.values() if r.split.partition == "test"}
    if train_groups & test_groups:
        raise RuntimeError("Loaded training and test data share a subject")
    branches = {}
    for branch, matrices in inputs.items():
        hyperparameters = dict(cfg["forest"])
        if scope == "smoke":
            hyperparameters["n_estimators"] = 12
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              RandomForestClassifier(**hyperparameters))
        model.fit(np.concatenate(matrices), index.true)
        if tuple(sorted(model.classes_)) != tuple(sorted(CLASSES)):
            raise ValueError(f"Training partition lacks a required class: {model.classes_}")
        dump(model, target / f"{branch}_model.joblib", compress=3)
        branches[branch] = model
        print(f"Fit {branch} on {len(index):,} reference beats in {len(train_groups)} subjects", flush=True)

    predictions, detection, noise_log, detection_edges = [], [], [], []
    for rid, record in records.items():
        if record.split.partition != "test":
            continue
        ref = references(record, cfg)
        predictions.extend(reference_predictions(record, ref, cfg, branches, record.signal, "clean", "none"))
        for kind in ("white", "drift"):
            noise = unit_noise(record, kind, cfg["noise"]["seed"], cfg["noise"]["drift_hz"])
            for db in cfg["noise"]["snr_db"]:
                perturbed, actual = inject(record.signal, noise, db)
                if not np.isclose(actual, db, atol=1e-9):
                    raise RuntimeError(f"SNR calibration failed for {rid} at {db} dB")
                predictions.extend(reference_predictions(record, ref, cfg, branches, perturbed, kind, str(db)))
                noise_log.append(dict(record_id=rid, noise=kind, target_db=db, actual_db=actual))
        filtered = bandpass(record.signal, record.fs, cfg["filter"])
        events, skips = detector_events(record, ref, filtered, branches["filtered"], cfg)
        detection.append(events)
        detection_edges.append(dict(record_id=rid, detections_without_window=skips))
        print(f"Evaluated {rid} ({len(ref.centres):,} QRS references)", flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    events = pd.concat(detection, ignore_index=True)
    pred.to_csv(target / "reference_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    events.to_csv(target / "detection_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(noise_log).to_csv(target / "snr_audit.csv", index=False, float_format="%.10f")
    pd.DataFrame(detection_edges).to_csv(target / "detection_window_skips.csv", index=False)
    summarize(pred, events, cfg, target)
    case = select_case(events, records)
    record = records[case.record_id]
    save_segment(target, "failure", record, bandpass(record.signal, record.fs, cfg["filter"]),
                 references(record, cfg), events, case)
    (target / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (target / "split.csv").write_bytes(Path(cfg["dataset"]["split"]).read_bytes())
    (target / "run_metadata.json").write_text(json.dumps(dict(
        scope=scope, python=platform.python_version(), system=platform.platform(),
        dependencies={pkg: version(pkg) for pkg in
                      ("numpy", "scipy", "scikit-learn", "wfdb", "pandas", "matplotlib", "reportlab", "pymupdf")},
        elapsed_seconds=round(time.perf_counter() - started, 3),
        peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        training_records=index.record_id.nunique(), training_subjects=len(train_groups),
        training_beats=len(index), test_subjects=len(test_groups),
        prediction_rows=len(pred), detection_rows=len(events), official_hashes=len(checksums),
        feature_names=list(FEATURE_NAMES), forest_trees=branches["raw"][-1].n_estimators,
        lock_sha256=sha256(Path("uv.lock")),
    ), indent=2) + "\n")
