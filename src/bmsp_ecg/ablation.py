"""Exploratory feature ablation on clean, filtered, reference-position ECG."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .data import load_record, sha256
from .experiment import make_forest, references
from .metrics import ENCODE, confusion, get_counts, macro_f1, ratio
from .signal import CLASSES, FEATURE_NAMES, MORPH_NAMES, RR_NAMES, bandpass, feature_matrix
from .stress import check_config

FEATURE_SETS = {"rr": RR_NAMES, "waveform": MORPH_NAMES, "all": FEATURE_NAMES}
KEY = ["record_id", "subject_group", "sample", "true"]
KEY_TYPES = {"record_id": str, "subject_group": str}
SOURCE_FILES = ("experiment.yaml", "split.csv", "data_audit.csv", "training_beats.csv.gz",
                "reference_predictions.csv.gz")


def checked_ablation(run):
    folder = run / "ablation"
    if not folder.exists():
        return None
    provenance = json.loads((folder / "provenance.json").read_text())
    for name in SOURCE_FILES:
        if sha256(run / name) != provenance["baseline_sha256"][name]:
            raise ValueError(f"Stale feature ablation for {name}; repeat the ablate command.")
    return folder


def run_ablation(cfg, root, split, checksums, run):
    check_config(cfg, run)
    digests = {name: sha256(run / name) for name in SOURCE_FILES}
    baseline = pd.read_csv(run / "data_audit.csv", dtype=KEY_TYPES, keep_default_na=False)
    cohort = baseline.loc[baseline.status == "included"].set_index("record_id")
    training = pd.read_csv(run / "training_beats.csv.gz", dtype=KEY_TYPES)
    reference = pd.read_csv(run / "reference_predictions.csv.gz", dtype=KEY_TYPES)
    full = reference.query("noise == 'clean' and branch == 'filtered'")[KEY + ["pred"]].reset_index(drop=True)

    indices = {"train": [], "test": []}
    blocks = {"train": [], "test": []}
    for item in split:
        rid = item.record_id
        if rid not in cohort.index:
            continue
        for suffix in ("hea", "dat", "atr"):
            expected = cohort.loc[rid, f"{suffix}_sha256"]
            name = f"{rid}.{suffix}"
            if expected and expected != (checksums.get(name) or sha256(root / name)):
                raise ValueError(f"{name} differs from the original run.")
        record, _ = load_record(root, item, cfg["dataset"], checksums)
        if record is None:
            raise ValueError(f"Missing saved study lead for {rid}")
        ref = references(record, cfg)
        signal = bandpass(record.signal, record.fs, cfg["filter"])
        features, valid = feature_matrix(signal, ref.centres, record.fs, cfg["window_s"])
        labels = ref.labels[valid]
        keep = np.isin(labels, CLASSES)
        indices[item.partition].append(pd.DataFrame(dict(
            record_id=rid, subject_group=item.subject_group,
            sample=ref.centres[valid][keep], true=labels[keep],
        )))
        blocks[item.partition].append(features[keep])

    train_index, test_index = (pd.concat(indices[part], ignore_index=True) for part in ("train", "test"))
    if not train_index.equals(training[KEY]) or not test_index.equals(full[KEY]):
        raise ValueError("Ablation beat indices differ from the original train or test ledger.")
    x_train, x_test = (np.concatenate(blocks[part]) for part in ("train", "test"))
    predictions = []
    for feature_set, columns in FEATURE_SETS.items():
        if feature_set == "all":
            frame = full
        else:
            select = [FEATURE_NAMES.index(name) for name in columns]
            model = make_forest(cfg["forest"])
            model.fit(x_train[:, select], train_index["true"])
            frame = test_index.assign(pred=model.predict(x_test[:, select]))
        predictions.append(frame.assign(feature_set=feature_set))
        print(f"Ablation {feature_set}: {len(frame):,} held-out predictions, {len(columns)} features", flush=True)

    events = pd.concat(predictions, ignore_index=True)
    rows = []
    for feature_set, frame in events.groupby("feature_set", sort=False):
        matrix = confusion(frame)
        tp, support, assigned = get_counts(matrix)
        s, n = ENCODE["S"], ENCODE["N"]
        rows.append(dict(
            feature_set=feature_set, features=len(FEATURE_SETS[feature_set]), beats=len(frame),
            macro_f1=float(macro_f1(matrix)), s_precision=float(ratio(tp[s], assigned[s])),
            s_recall=float(ratio(tp[s], support[s])), s_to_n=int(matrix[s, n]),
        ))
    target = run / "ablation"
    with TemporaryDirectory(prefix=".ablation-", dir=run) as directory:
        staged = Path(directory)
        events.to_csv(staged / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(rows).to_csv(staged / "metrics.csv", index=False, float_format="%.10f")
        provenance = dict(
            baseline_sha256=digests, feature_sets=FEATURE_SETS, forest=cfg["forest"],
            noise="clean", positions="reference", branch="filtered", all_predictions="reference_predictions.csv.gz",
        )
        (staged / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if any(sha256(run / name) != digest for name, digest in digests.items()):
            raise ValueError("The baseline run changed during the ablation.")
        target.mkdir(exist_ok=True)
        for file in staged.iterdir():
            file.replace(target / file.name)
    print(f"Saved exploratory feature ablation in {target}.", flush=True)
