"""Independently recompute reported metrics and check the saved beat ledgers."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from bmsp_ecg.stress import checked_stress

CLASSES = ["N", "S", "V", "F"]
AXIS = [*CLASSES, "none"]
INT_DTYPE = {"record_id": str, "subject_group": str}


def read(root, name, **kwargs):
    return pd.read_csv(root / name, **kwargs)


def check_metrics(name, rows, summary, classes, matrices):
    actual, predicted = rows["true"], rows["pred"]
    matrix = confusion_matrix(actual, predicted, labels=AXIS)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0,
    )
    file_class = classes.loc[name].set_index("label").reindex(CLASSES)
    file_matrix = matrices.loc[name].pivot(index="true", columns="pred", values="count").loc[AXIS, AXIS]
    np.testing.assert_array_equal(file_matrix, matrix)
    assigned = matrix[:, :4].sum(axis=0)
    np.testing.assert_array_equal(file_class.support, support)
    np.testing.assert_array_equal(file_class.tp, matrix.diagonal()[:4])
    np.testing.assert_array_equal(file_class.fp, assigned - matrix.diagonal()[:4])
    np.testing.assert_array_equal(file_class.fn, support - matrix.diagonal()[:4])
    np.testing.assert_allclose(file_class[["precision", "recall", "f1"]], np.column_stack((precision, recall, f1)),
                               rtol=0, atol=1e-9)
    if name in summary.index:
        assert summary.loc[name, "support"] == support.sum()
        assert np.isclose(summary.loc[name, "macro_f1"], f1.mean(), rtol=0, atol=1e-9)
        if name != "end_to_end":
            assert np.isclose(summary.loc[name, "accuracy"], (actual == predicted).mean(), rtol=0, atol=1e-9)
    return float(f1.mean())


def check_interval(before, after, expected, draws, groups):
    scores = []
    for rows in (before, after):
        matrices = [confusion_matrix(block["true"], block["pred"], labels=AXIS)
                    for group in groups if len(block := rows.loc[rows.subject_group == group])]
        assert len(matrices) == len(groups)
        merged = np.tensordot(draws, np.array(matrices), axes=1)
        diag = merged[:, :4, :4].diagonal(axis1=1, axis2=2)
        den = merged[:, :4, :].sum(axis=2) + merged[:, :, :4].sum(axis=1)
        per_class = np.divide(2 * diag, den, out=np.zeros_like(den, dtype=float), where=den > 0)
        scores.append(per_class.mean(axis=1))
    measured = np.quantile(scores[1] - scores[0], [0.025, 0.975])
    np.testing.assert_allclose(measured, [expected.lo, expected.hi], rtol=0, atol=1e-9)
    assert expected.subjects == len(groups)
    assert expected.bootstrap_repetitions == len(draws)


def check_posthoc(root, reference, events):
    table = read(root, "sensitivity_record_213.csv", keep_default_na=False).set_index("scope")
    assert list(table.index) == ["all", "without_213"]
    for scope, omitted in (("all", ""), ("without_213", "213")):
        raw = reference["raw"].loc[reference["raw"].record_id != omitted]
        filtered = reference["filtered"].loc[reference["filtered"].record_id != omitted]
        row = table.loc[scope]
        assert row.removed_record == omitted
        assert row.beats == len(raw) == len(filtered)
        assert row.fusion_beats == raw["true"].eq("F").sum()
        scores = [f1_score(block["true"], block.pred, labels=CLASSES, average="macro", zero_division=0)
                  for block in (raw, filtered)]
        np.testing.assert_allclose([row.raw_f1, row.filtered_f1, row.delta],
                                   [*scores, scores[1] - scores[0]], rtol=0, atol=1e-9)
    errors = read(root, "minority_errors.csv").set_index(["measure", "true", "outcome"])
    assert len(errors) == 4
    for measure, actual, outcome in (
        ("reference_filtered_classification", "S", "S"),
        ("reference_filtered_classification", "S", "N"),
        ("xqrs_detection_without_added_noise", "V", "miss"),
        ("xqrs_detection_without_added_noise", "N", "miss"),
    ):
        frame, outcome_field = (events, "status") if outcome == "miss" else (reference["filtered"], "pred")
        block = frame.loc[frame["true"] == actual]
        row = errors.loc[(measure, actual, outcome)]
        count = block[outcome_field].eq(outcome).sum()
        assert row["count"] == count and row.support == len(block)
        assert np.isclose(row.rate, count / len(block), rtol=0, atol=1e-9)
    print("Verified the record 213 sensitivity scores and minority-class error counts.")


def check_detection(ledger, tolerance):
    counts = ledger.status.value_counts()
    tp, fp, fn = [int(counts.get(key, 0)) for key in ("matched", "false_positive", "miss")]
    assert tp + fp + fn == len(ledger)
    assert not ledger.dropna(subset=["ref_sample"]).duplicated(["record_id", "ref_sample"]).any()
    assert not ledger.dropna(subset=["det_sample"]).duplicated(["record_id", "det_sample"]).any()
    matched = ledger[ledger.status == "matched"]
    shift = matched.det_sample - matched.ref_sample
    assert np.all(shift.abs() <= tolerance) and np.array_equal(shift, matched.shift_samples)
    return tp, fp, fn, matched, shift


def check_stress(root, events, tolerance, snr, draws, groups):
    folder = checked_stress(root)
    if folder is None:
        print("0 dB drift detector replay absent; use the stress command to generate it.")
        return
    metadata = json.loads((folder / "provenance.json").read_text())
    assert metadata["noise"] == "drift" and metadata["target_db"] == 0
    ledger = read(folder, "events.csv.gz", dtype=INT_DTYPE)
    metrics = read(folder, "metrics.csv").set_index("branch")
    audit = read(folder, "noise_audit.csv", dtype={"record_id": str})
    assert not audit.duplicated(["record_id", "branch"]).any()
    source = snr.query("noise == 'drift' and target_db == 0").copy()
    source["record_id"] = source.record_id.astype(str)
    paired_snr = audit.merge(source[["record_id", "actual_db"]], on="record_id", validate="many_to_one")
    assert len(paired_snr) == len(audit) == len(source) * 2
    np.testing.assert_allclose(paired_snr.actual_db_x, paired_snr.actual_db_y, rtol=0, atol=1e-9)
    assert audit.shared_input_sha256.str.fullmatch(r"[0-9a-f]{64}").all()
    hashes = audit.pivot(index="record_id", columns="branch", values="shared_input_sha256")
    assert hashes.raw.notna().all() and hashes.raw.equals(hashes.filtered)

    key = ["record_id", "ref_sample", "true"]
    reference = events.dropna(subset=["ref_sample"])[key].sort_values(key, ignore_index=True)
    scored, scores = {}, {}
    assert set(ledger.branch) == set(metrics.index) == {"raw", "filtered"}
    for branch in ("raw", "filtered"):
        block = ledger[ledger.branch == branch]
        actual = block.dropna(subset=["ref_sample"])[key].sort_values(key, ignore_index=True)
        pd.testing.assert_frame_equal(reference, actual)
        tp, fp, fn, _, _ = check_detection(block, tolerance)
        end = block.loc[block["true"].isin(CLASSES) | (block.status == "false_positive")]
        _, recall, f1, support = precision_recall_fscore_support(
            end["true"], end.pred, labels=CLASSES, zero_division=0,
        )
        row = metrics.loc[branch]
        assert (row.support, row.matched, row.missed, row.extra) == (support.sum(), tp, fn, fp)
        np.testing.assert_allclose([row.qrs_f1, row.macro_f1, row.s_recall, row.v_recall],
                                   [2 * tp / (2 * tp + fn + fp), f1.mean(), recall[1], recall[2]],
                                   rtol=0, atol=1e-9)
        scored[branch], scores[branch] = end, f1.mean()
    delta = read(folder, "paired_delta.csv").iloc[0]
    assert np.isclose(delta.delta, scores["filtered"] - scores["raw"], rtol=0, atol=1e-9)
    check_interval(scored["raw"], scored["filtered"], delta, draws, groups)
    print(f"Verified {len(ledger):,} additional detection rows and the 0 dB drift results.")


def validate(root):
    config = yaml.safe_load((root / "experiment.yaml").read_text())
    meta = json.loads((root / "run_metadata.json").read_text())
    assert meta["scope"] == "run"
    summary = read(root, "summary.csv").set_index("protocol")
    per_class = read(root, "per_class.csv").set_index("protocol")
    matrices = read(root, "confusion.csv", keep_default_na=False).set_index("protocol")
    predictions = read(root, "reference_predictions.csv.gz", dtype={**INT_DTYPE, "snr_db": str})
    training = read(root, "training_beats.csv.gz", dtype=INT_DTYPE)
    events = read(root, "detection_events.csv.gz", dtype=INT_DTYPE)
    manifest = read(root, "split.csv", dtype=INT_DTYPE)
    support = read(root, "beat_counts.csv", dtype={"record_id": str})
    data_audit = read(root, "data_audit.csv", dtype={"record_id": str})
    eligible = data_audit.query("status == 'included'").record_id
    side = manifest.set_index("record_id").partition
    assert set(training.record_id) == {rid for rid in eligible if side[rid] == "train"}
    assert set(predictions.record_id) == {rid for rid in eligible if side[rid] == "test"}
    assert not set(training.subject_group) & set(predictions.subject_group)
    assert len(training) == meta["training_beats"]
    assert len(predictions) == meta["prediction_rows"] and len(events) == meta["detection_rows"]
    assert not training.duplicated(["record_id", "sample"]).any()
    assert not predictions.duplicated(["record_id", "sample", "noise", "snr_db", "branch"]).any()

    groups = sorted(predictions.subject_group.unique())
    samples = config["bootstrap"]["repetitions"]
    draws = np.random.default_rng(config["bootstrap"]["seed"]).multinomial(
        len(groups), np.repeat(1 / len(groups), len(groups)), size=samples,
    )
    noise = read(root, "noise_metrics.csv", dtype={"snr_db": str}).set_index(["noise", "snr_db"])
    computed = {}
    keys = ["record_id", "sample", "true"]
    for (kind, db), condition in predictions.groupby(["noise", "snr_db"]):
        branches = {branch: rows for branch, rows in condition.groupby("branch")}
        scores = {}
        for branch, rows in branches.items():
            label = f"reference_{branch}" if kind == "clean" else f"{kind}_{db}_{branch}"
            scores[branch] = check_metrics(label, rows, summary, per_class, matrices)
            expected = support.query("partition == 'test'")[CLASSES].sum().to_dict()
            assert rows["true"].value_counts().to_dict() == expected
        raw, filt = branches["raw"], branches["filtered"]
        assert raw[keys].reset_index(drop=True).equals(filt[keys].reset_index(drop=True))
        if kind == "clean":
            assert branches["all_n"].pred.eq("N").all()
            computed["raw"], computed["filtered"] = raw, filt
        row = noise.loc[(kind, db)]
        np.testing.assert_allclose([row.raw_f1, row.filtered_f1, row.delta],
                                   [scores["raw"], scores["filtered"], scores["filtered"] - scores["raw"]],
                                   rtol=0, atol=1e-9)
        assert row.support == len(raw)
        check_interval(raw, filt, row, draws, groups)
    assert set(noise.index) == {("clean", "none")} | {
        (kind, str(db)) for kind in ("white", "drift") for db in config["noise"]["snr_db"]
    }

    snr = read(root, "snr_audit.csv")
    assert len(snr) == len(groups) * 2 * len(config["noise"]["snr_db"])
    assert np.max(np.abs(snr.actual_db - snr.target_db)) <= 1e-9
    assert not snr.duplicated(["record_id", "noise", "target_db"]).any()

    qrs = read(root, "detection_by_record.csv", dtype={"record_id": str}).set_index("record_id")
    assert set(qrs.index) == set(events.record_id) | {"ALL"}
    fs = config["dataset"]["fs_hz"]
    tolerance = round(fs * config["detection"]["tolerance_ms"] / 1000)
    for rid, block in [("ALL", events), *events.groupby("record_id")]:
        n_tp, n_fp, n_fn, matched, shift = check_detection(block, tolerance)
        for key, value in zip(("tp", "fp", "fn", "matched_excluded"), (
            n_tp, n_fp, n_fn, (~matched["true"].isin(CLASSES)).sum(),
        )):
            assert qrs.loc[rid, key] == value
        np.testing.assert_allclose(
            qrs.loc[rid, ["precision", "recall", "f1", "median_abs_ms", "p95_abs_ms"]].to_numpy(dtype=float),
            [n_tp / (n_tp + n_fp), n_tp / (n_tp + n_fn), 2 * n_tp / (2 * n_tp + n_fn + n_fp),
             shift.abs().median() * 1000 / fs, shift.abs().quantile(.95) * 1000 / fs],
            rtol=0, atol=1e-9,
        )

    matched = events.loc[(events.status == "matched") & events["true"].isin(CLASSES)]
    end = events.loc[events["true"].isin(CLASSES) | (events.status == "false_positive")]
    ref_matches = computed["filtered"].merge(
        matched[["record_id", "ref_sample"]].rename(columns={"ref_sample": "sample"}),
        on=["record_id", "sample"], validate="one_to_one",
    )
    for name, rows in (("matched_classification", matched), ("end_to_end", end),
                       ("reference_on_matches", ref_matches)):
        check_metrics(name, rows, summary, per_class, matrices)

    paired = read(root, "paired_detection.csv").set_index("comparison")
    for name, before, after in (
        ("reference_raw_to_filtered", computed["raw"], computed["filtered"]),
        ("same_beats_reference_to_matched", ref_matches, matched),
        ("reference_to_end_to_end", computed["filtered"], end),
    ):
        row = paired.loc[name]
        scores = [f1_score(rows["true"], rows["pred"], labels=CLASSES, average="macro", zero_division=0)
                  for rows in (before, after)]
        assert np.isclose(row.delta, scores[1] - scores[0], rtol=0, atol=1e-9)
        check_interval(before, after, row, draws, groups)

    case = json.loads((root / "failure_meta.json").read_text())
    focus = events[(events.record_id == case["record_id"]) & (events.ref_sample == case["focus_sample"])]
    assert len(focus) == 1 and focus.iloc[0].status == case["focus_status"]
    assert case["start"] < case["focus_sample"] < case["stop"]
    check_posthoc(root, computed, events)
    check_stress(root, events, tolerance, snr, draws, groups)
    print(f"Verified {len(training):,} training indices, {len(predictions):,} classifications, "
          f"{len(events):,} detection rows, {len(noise)} paired noise settings and all metrics.")
    return summary


def compare(first, second):
    names = [
        "data_audit.csv", "beat_counts.csv", "training_beats.csv.gz", "reference_predictions.csv.gz",
        "detection_events.csv.gz", "summary.csv", "per_class.csv", "confusion.csv", "noise_metrics.csv",
        "paired_detection.csv", "detection_by_record.csv", "detection_window_skips.csv", "snr_audit.csv",
        "qc_waveform.csv", "failure_waveform.csv", "failure_meta.json", "experiment.yaml", "split.csv",
        "sensitivity_record_213.csv", "minority_errors.csv",
    ]
    if (first / "stress").is_dir() or (second / "stress").is_dir():
        names += ["stress/events.csv.gz", "stress/noise_audit.csv", "stress/metrics.csv", "stress/paired_delta.csv"]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    print(f"Replayed {len(names)} files byte-for-byte, including beat indices and per-beat predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    validate(args.run)
    if args.compare:
        validate(args.compare)
        compare(args.run, args.compare)
