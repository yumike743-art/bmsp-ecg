"""Replay the trained models under shared 0 dB drift and refresh post-hoc tables."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from joblib import load

from .data import load_record, sha256
from .experiment import detector_events, references
from .metrics import confusion, detector_rows, get_counts, macro_f1, paired_bootstrap, ratio, split_detection, write_posthoc
from .signal import CLASSES, bandpass, inject, unit_noise

BRANCHES = ("raw", "filtered")
STRESS_SNR_DB = 0
BASELINE_FILES = ("experiment.yaml", "split.csv", "reference_predictions.csv.gz", "detection_events.csv.gz")
MODEL_FILES = tuple(f"{branch}_model.joblib" for branch in BRANCHES)


def check_config(cfg, run):
    if yaml.safe_load((run / "experiment.yaml").read_text()) != cfg:
        raise ValueError("Active configuration differs from the saved full run.")
    if Path(cfg["dataset"]["split"]).read_bytes() != (run / "split.csv").read_bytes():
        raise ValueError("The current subject manifest differs from the saved full run.")


def rebuild_posthoc(cfg, run):
    check_config(cfg, run)
    preds = pd.read_csv(run / "reference_predictions.csv.gz", dtype={"record_id": str})
    events = pd.read_csv(run / "detection_events.csv.gz", dtype={"record_id": str})
    with TemporaryDirectory(prefix=".posthoc-", dir=run) as directory:
        staged = Path(directory)
        write_posthoc(preds, events, staged)
        for file in staged.iterdir():
            file.replace(run / file.name)
        stress = checked_stress(run)
        if stress is not None:
            write_alignment(pd.read_csv(stress / "events.csv.gz", dtype={"record_id": str}), staged)
            (staged / "alignment.csv").replace(stress / "alignment.csv")
    print(f"Recomputed sensitivity and minority-error tables in {run}; no training.", flush=True)


def checked_stress(run):
    folder = run / "stress"
    if not folder.exists():
        return None
    metadata = json.loads((folder / "provenance.json").read_text())
    for name in BASELINE_FILES:
        if sha256(run / name) != metadata["baseline_sha256"][name]:
            raise ValueError(f"Stale drift results for {name}; repeat the stress command.")
    for name in MODEL_FILES:
        if (run / name).is_file() and sha256(run / name) != metadata["model_sha256"][name]:
            raise ValueError(f"Stale drift results for {name}; repeat the stress command.")
    return folder


def write_alignment(events, target):
    key = ["record_id", "ref_sample"]
    fields = key + ["status", "det_sample", "pred"]
    ref = events.dropna(subset=["ref_sample"])
    raw, filtered = (ref.loc[ref.branch == branch, fields] for branch in BRANCHES)
    paired = raw.merge(filtered, on=key, how="outer", validate="one_to_one",
                       indicator=True, suffixes=("_raw", "_filtered"))
    if not paired["_merge"].eq("both").all():
        raise ValueError("Drift branches contain different reference beats.")
    matched = paired.loc[paired.status_raw.eq("matched") & paired.status_filtered.eq("matched")]
    moved = matched.det_sample_raw.ne(matched.det_sample_filtered)
    row = dict(
        references=len(paired), status_disagreements=int(paired.status_raw.ne(paired.status_filtered).sum()),
        common_matches=len(matched), position_changes=int(moved.sum()),
        position_change_rate=float(ratio(moved.sum(), len(matched))),
        same_position_prediction_changes=int((~moved & matched.pred_raw.ne(matched.pred_filtered)).sum()),
    )
    pd.DataFrame([row]).to_csv(target / "alignment.csv", index=False, float_format="%.10f")


def summarize_stress(events, cfg, staged):
    end_to_end, rows = {}, []
    for branch in BRANCHES:
        block = events.loc[events.branch == branch]
        _, scored = split_detection(block)
        end_to_end[branch] = scored
        matrix = confusion(scored)
        tp, support, _ = get_counts(matrix)
        recall = dict(zip(CLASSES, ratio(tp, support)))
        qrs = detector_rows(block, cfg["dataset"]["fs_hz"]).iloc[0]
        rows.append(dict(
            branch=branch, support=int(support.sum()), qrs_f1=qrs.f1,
            matched=qrs.tp, missed=qrs.fn, extra=qrs.fp, macro_f1=float(macro_f1(matrix)),
            s_recall=recall["S"], v_recall=recall["V"],
        ))
    pd.DataFrame(rows).to_csv(staged / "metrics.csv", index=False, float_format="%.10f")
    bootstrap = paired_bootstrap(
        end_to_end["raw"], end_to_end["filtered"],
        cfg["bootstrap"]["repetitions"], cfg["bootstrap"]["seed"],
    )
    pd.DataFrame([bootstrap]).to_csv(staged / "paired_delta.csv", index=False, float_format="%.10f")
    write_alignment(events, staged)


def run_drift_stress(cfg, data_root, split, checksums, run):
    check_config(cfg, run)
    base_hashes = {name: sha256(run / name) for name in BASELINE_FILES}
    model_hashes = {name: sha256(run / name) for name in MODEL_FILES}
    models = {branch: load(run / name) for branch, name in zip(BRANCHES, MODEL_FILES)}
    baseline = pd.read_csv(run / "data_audit.csv", dtype={"record_id": str}, keep_default_na=False)
    cohort = baseline.query("partition == 'test' and status == 'included'").set_index("record_id")
    source_snr = pd.read_csv(run / "snr_audit.csv", dtype={"record_id": str})
    source_snr = source_snr.query("noise == 'drift' and target_db == @STRESS_SNR_DB").set_index("record_id")
    blocks, noise_audit = [], []
    for item in split:
        rid = item.record_id
        if rid not in cohort.index:
            continue
        for suffix in ("hea", "dat", "atr"):
            expected = cohort.loc[rid, suffix + "_sha256"]
            file = f"{rid}.{suffix}"
            if expected and expected != (checksums.get(file) or sha256(data_root / file)):
                raise ValueError(f"{file} differs from the data used to fit and evaluate the main run.")
        record, _ = load_record(data_root, item, cfg["dataset"], checksums)
        if record is None:
            raise ValueError(f"Missing saved test lead for {rid}")
        ref = references(record, cfg)
        noise = unit_noise(record, "drift", cfg["noise"]["seed"], cfg["noise"]["drift_hz"])
        perturbed, achieved = inject(record.signal, noise, STRESS_SNR_DB)
        if not np.isclose(achieved, source_snr.loc[rid, "actual_db"], rtol=0, atol=1e-9):
            raise ValueError(f"Drift SNR mismatch against the reference-position run for {rid}")
        input_hash = hashlib.sha256(perturbed.tobytes()).hexdigest()
        sources = {"raw": perturbed, "filtered": bandpass(perturbed, record.fs, cfg["filter"])}
        for branch, waveform in sources.items():
            block, skipped = detector_events(record, ref, waveform, models[branch], cfg)
            blocks.append(block.assign(branch=branch))
            noise_audit.append(dict(
                record_id=rid, branch=branch, actual_db=achieved, window_skips=skipped,
                shared_input_sha256=input_hash,
            ))
        print(f"0 dB drift: both branches evaluated on {rid}", flush=True)
    events = pd.concat(blocks, ignore_index=True)
    target = run / "stress"
    with TemporaryDirectory(prefix=".stress-", dir=run) as directory:
        staged = Path(directory)
        events.to_csv(staged / "events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(noise_audit).to_csv(staged / "noise_audit.csv", index=False, float_format="%.10f")
        summarize_stress(events, cfg, staged)
        provenance = dict(
            noise="drift", target_db=STRESS_SNR_DB, drift_hz=cfg["noise"]["drift_hz"],
            baseline_sha256=base_hashes, model_sha256=model_hashes,
        )
        if any(sha256(run / name) != digest for name, digest in (base_hashes | model_hashes).items()):
            raise ValueError("The baseline run changed during the drift comparison.")
        target.mkdir(exist_ok=True)
        for file in staged.iterdir():
            file.replace(target / file.name)
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Finished 0 dB drift comparisons in {target}; no training.", flush=True)
