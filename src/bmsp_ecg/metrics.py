"""Metrics and paired subject-bootstrap intervals from saved event ledgers."""

import numpy as np
import pandas as pd

from .signal import CLASSES

AXIS = (*CLASSES, "none")
ENCODE = {name: index for index, name in enumerate(AXIS)}
SENSITIVITY_RECORD = "213"


def confusion(frame):
    actual = frame["true"].map(ENCODE).to_numpy(dtype=int)
    predicted = frame["pred"].map(ENCODE).to_numpy(dtype=int)
    return np.bincount(actual * 5 + predicted, minlength=25).reshape(5, 5)


def ratio(num, den):
    return np.divide(num, den, out=np.zeros(np.shape(den), dtype=float), where=den != 0)


def get_counts(matrix):
    tp = np.diagonal(matrix[..., :4, :4], axis1=-2, axis2=-1)
    support = matrix[..., :4, :].sum(axis=-1)
    assigned = matrix[..., :, :4].sum(axis=-2)
    return tp, support, assigned


def macro_f1(matrix):
    tp, support, assigned = get_counts(matrix)
    return ratio(2 * tp, support + assigned).mean(axis=-1)


def classification_rows(name, frame, file_rows, confusion_rows):
    matrix = confusion(frame)
    tp, support, assigned = get_counts(matrix)
    precision, recall, f1 = ratio(tp, assigned), ratio(tp, support), ratio(2 * tp, support + assigned)
    for i, label in enumerate(CLASSES):
        file_rows.append(dict(protocol=name, label=label, precision=precision[i], recall=recall[i],
                              f1=f1[i], support=support[i], tp=tp[i], fp=assigned[i] - tp[i],
                              fn=support[i] - tp[i]))
    for i, actual in enumerate(AXIS):
        for j, pred in enumerate(AXIS):
            confusion_rows.append(dict(protocol=name, true=actual, pred=pred, count=matrix[i, j]))
    return dict(protocol=name, macro_f1=float(f1.mean()), support=int(support.sum()),
                accuracy=float(tp.sum() / support.sum()) if support.sum() and "end_to_end" != name else np.nan)


def split_detection(events):
    matched = events["status"] == "matched"
    included = events["true"].isin(CLASSES)
    detect_only = events["status"] == "false_positive"
    normalized = events.assign(true=events["true"].where(~detect_only, "none"),
                               pred=events["pred"].where(events["status"] != "miss", "none"))
    return normalized[matched & included], normalized[included | detect_only]


def detector_rows(events, fs):
    rows = []
    for name, group in [("ALL", events), *events.groupby("record_id", sort=True)]:
        counts = group["status"].value_counts()
        tp, fp, fn = (int(counts.get(status, 0)) for status in ("matched", "false_positive", "miss"))
        error_ms = group.loc[group["status"] == "matched", "shift_samples"].abs() * 1000 / fs
        rows.append(dict(record_id=name, tp=tp, fp=fp, fn=fn, precision=float(ratio(tp, tp + fp)),
                         recall=float(ratio(tp, tp + fn)), f1=float(ratio(2 * tp, 2 * tp + fp + fn)),
                         matched_excluded=int(((group["status"] == "matched") &
                                              ~group["true"].isin(CLASSES)).sum()),
                         median_abs_ms=error_ms.median(), p95_abs_ms=error_ms.quantile(.95)))
    return pd.DataFrame(rows)


def paired_bootstrap(left, right, repetitions, seed):
    """Paired test-subject resampling. Both arms share each bootstrap draw."""
    groups = sorted(set(left["subject_group"]) | set(right["subject_group"]))
    batches = [
        np.stack([confusion(frame[frame["subject_group"] == group]) for group in groups])
        for frame in (left, right)
    ]
    size = len(groups)
    counts = np.random.default_rng(seed).multinomial(size, np.full(size, 1 / size), size=repetitions)
    scores = [macro_f1(np.einsum("bs,sij->bij", counts, batch)) for batch in batches]
    paired = scores[1] - scores[0]
    return dict(delta=float(macro_f1(batches[1].sum(0)) - macro_f1(batches[0].sum(0))),
                lo=float(np.quantile(paired, .025)), hi=float(np.quantile(paired, .975)),
                subjects=size, bootstrap_repetitions=repetitions)


def write_posthoc(predictions, events, target):
    clean = predictions.loc[predictions.noise == "clean"]
    by_scope = []
    for scope, block in (
        ("all", clean),
        ("without_213", clean.loc[clean.record_id != SENSITIVITY_RECORD]),
    ):
        raw = block.loc[block.branch == "raw"]
        filtered = block.loc[block.branch == "filtered"]
        scores = [float(macro_f1(confusion(arm))) for arm in (raw, filtered)]
        by_scope.append(dict(
            scope=scope, removed_record="" if scope == "all" else SENSITIVITY_RECORD,
            beats=len(raw), fusion_beats=int(raw["true"].eq("F").sum()),
            raw_f1=scores[0], filtered_f1=scores[1], delta=scores[1] - scores[0],
        ))
    pd.DataFrame(by_scope).to_csv(
        target / "sensitivity_record_213.csv", index=False, float_format="%.10f",
    )

    filtered_s = clean.loc[(clean.branch == "filtered") & (clean["true"] == "S")]
    counts = []
    for label in ("S", "N"):
        counts.append(dict(
            measure="reference_filtered_classification", true="S", outcome=label,
            count=int(filtered_s.pred.eq(label).sum()), support=len(filtered_s),
        ))
    for label in ("V", "N"):
        reference = events.loc[events["true"] == label]
        counts.append(dict(
            measure="xqrs_detection_without_added_noise", true=label, outcome="miss",
            count=int(reference.status.eq("miss").sum()), support=len(reference),
        ))
    errors = pd.DataFrame(counts)
    errors["rate"] = ratio(errors["count"].to_numpy(), errors["support"].to_numpy())
    errors.to_csv(target / "minority_errors.csv", index=False, float_format="%.10f")


def summarize(predictions, events, cfg, target):
    rows, class_rows, matrix_rows, noise_rows = [], [], [], []
    for (noise, db), cell in predictions.groupby(["noise", "snr_db"], sort=False):
        arms = {}
        for branch, frame in cell.groupby("branch", sort=False):
            name = f"{noise}_{db}_{branch}"
            result = classification_rows(name, frame, class_rows, matrix_rows)
            if noise == "clean":
                rows.append({**result, "protocol": f"reference_{branch}"})
            if branch in {"raw", "filtered"}:
                arms[branch] = (frame, result)
        raw, raw_score = arms["raw"]
        filtered, filtered_score = arms["filtered"]
        if not raw[["record_id", "sample", "true"]].reset_index(drop=True).equals(
            filtered[["record_id", "sample", "true"]].reset_index(drop=True)
        ):
            raise ValueError(f"Unpaired reference beats at {noise}/{db}")
        comparison = paired_bootstrap(raw, filtered, cfg["bootstrap"]["repetitions"], cfg["bootstrap"]["seed"])
        noise_rows.append(dict(noise=noise, snr_db=db, raw_f1=raw_score["macro_f1"],
                               filtered_f1=filtered_score["macro_f1"], support=raw_score["support"],
                               **comparison))
    matched, end_to_end = split_detection(events)
    for name, frame in (("matched_classification", matched), ("end_to_end", end_to_end)):
        rows.append(classification_rows(name, frame, class_rows, matrix_rows))
    raw_ref = predictions.query("noise == 'clean' and branch == 'raw'")
    filt_ref = predictions.query("noise == 'clean' and branch == 'filtered'")
    reference_on_matches = filt_ref.merge(
        matched[["record_id", "ref_sample"]].rename(columns={"ref_sample": "sample"}),
        on=["record_id", "sample"], validate="one_to_one",
    )
    rows.append(classification_rows("reference_on_matches", reference_on_matches, class_rows, matrix_rows))
    paired_rows = []
    for comparison_name, left, right in (
        ("reference_raw_to_filtered", raw_ref, filt_ref),
        ("same_beats_reference_to_matched", reference_on_matches, matched),
        ("reference_to_end_to_end", filt_ref, end_to_end),
    ):
        result = paired_bootstrap(left, right, cfg["bootstrap"]["repetitions"], cfg["bootstrap"]["seed"])
        paired_rows.append({"comparison": comparison_name, **result})
    # Export stable protocol names for clean confusion and per-class rows.
    for row in class_rows + matrix_rows:
        if row["protocol"].startswith("clean_none_"):
            row["protocol"] = row["protocol"].replace("clean_none_", "reference_", 1)
    for name, values in (
        ("summary", rows), ("per_class", class_rows), ("confusion", matrix_rows),
        ("noise_metrics", noise_rows), ("paired_detection", paired_rows),
    ):
        pd.DataFrame(values).to_csv(target / f"{name}.csv", index=False, float_format="%.10f")
    detector_rows(events, cfg["dataset"]["fs_hz"]).to_csv(
        target / "detection_by_record.csv", index=False, float_format="%.10f"
    )
