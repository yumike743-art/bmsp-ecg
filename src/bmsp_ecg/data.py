"""Versioned acquisition, manifest validation and physical lead extraction."""

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import wfdb


@dataclass(frozen=True)
class SplitRow:
    record_id: str
    subject_group: str
    partition: str
    reason: str


def read_split(path):
    with path.open(newline="") as source:
        rows = [SplitRow(**row) for row in csv.DictReader(source)]
    by_id = {row.record_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate recording in split manifest")
    partitions = {side: [r for r in rows if r.partition == side]
                  for side in ("train", "test", "exclude")}
    if [len(partitions[k]) for k in partitions] != [23, 21, 4] or len(rows) != 48:
        raise ValueError("Expected 23 training, 21 test and four paced recordings")
    if {r.record_id for r in partitions["exclude"]} != {"102", "104", "107", "217"}:
        raise ValueError("Unexpected set of paced records")
    groups = {side: {r.subject_group for r in partitions[side]}
              for side in ("train", "test")}
    if groups["train"] & groups["test"]:
        raise ValueError("A subject appears across train and test")
    if len(groups["train"]) != 22 or len(groups["test"]) != 21:
        raise ValueError("Unexpected number of subject groups")
    if by_id["201"].subject_group != by_id["202"].subject_group or any(
        by_id[record].partition != "train" for record in ("201", "202")
    ):
        raise ValueError("Records 201 and 202 must share a training group")
    return rows


def listed_files(rows):
    return [f"{row.record_id}.{suffix}" for row in rows for suffix in ("hea", "dat", "atr")]


def read_checksums(path):
    return {name.removeprefix("./"): digest for digest, name in
            (line.split(maxsplit=1) for line in path.read_text().splitlines() if line.strip())}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_directory(root, rows, version):
    expected = listed_files(rows)
    absent = [name for name in expected if not (root / name).is_file()]
    if absent:
        raise FileNotFoundError(
            f"{root}: missing {len(absent)} MIT-BIH v{version} files (e.g. {absent[0]}). "
            "Run data fetch --config configs/experiment.yaml or provide --data-dir DIRECTORY."
        )
    verified = {}
    sums = root / "SHA256SUMS.txt"
    if sums.is_file():
        reference = read_checksums(sums)
        for name in expected:
            digest = sha256(root / name)
            if digest != reference.get(name):
                raise ValueError(f"Checksum mismatch: {root / name}. Run data fetch to repair.")
            verified[name] = digest
    return verified


def fetch_database(root, rows, version):
    """Stream official WFDB-format files; replace a file only after integrity checks."""
    base = f"https://physionet.org/files/mitdb/{version}/"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=root) as folder:
        staging = Path(folder)
        with urlopen(base + "SHA256SUMS.txt", timeout=45) as source:
            file = staging / "SHA256SUMS.txt"
            file.write_bytes(source.read())
            file.replace(root / "SHA256SUMS.txt")
        reference = read_checksums(root / "SHA256SUMS.txt")

        def fetch_one(name):
            target = root / name
            digest = reference[name]
            if target.is_file() and sha256(target) == digest:
                return
            staged = staging / name
            for attempt in range(3):
                try:
                    with urlopen(base + name, timeout=45) as source, staged.open("wb") as output:
                        while block := source.read(256 * 1024):
                            output.write(block)
                    if sha256(staged) != digest:
                        raise ValueError(f"Failed official SHA-256 check for {name}")
                    staged.replace(target)
                    return
                except (OSError, ValueError):
                    staged.unlink(missing_ok=True)
                    if attempt == 2:
                        raise

        # The database files are WFDB-native; targeted HTTPS transfers provide SHA checks
        # and atomic retries, while all headers, samples and annotations use WFDB readers.
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch_one, listed_files(rows)))
    return check_directory(root, rows, version)


def lead_column(names, target):
    hits = [index for index, name in enumerate(names) if name == target]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous lead name: {target}")
    return hits[0] if hits else None


@dataclass
class Record:
    split: SplitRow
    signal: np.ndarray
    samples: np.ndarray
    symbols: np.ndarray
    fs: int
    unit: str


def load_record(root, item, dataset, verified):
    base = str(root / item.record_id)
    header = wfdb.rdheader(base)
    ann = wfdb.rdann(base, "atr")
    column = lead_column(header.sig_name, dataset["lead"])
    considered = item.partition != "exclude" and column is not None
    if header.fs != dataset["fs_hz"] or header.n_sig != 2:
        raise ValueError(f"Unexpected acquisition in {item.record_id}: {header.fs} Hz, {header.n_sig} leads")
    if np.any((ann.sample < 0) | (ann.sample >= header.sig_len)):
        raise ValueError(f"Annotation out of bounds in {item.record_id}")
    annotation_counts = Counter(ann.symbol)
    digest_columns = {suffix + "_sha256": verified.get(f"{item.record_id}.{suffix}", "")
                      for suffix in ("hea", "atr", "dat")}
    audit = {
        **vars(item), "status": ("included" if considered else "paced_record"
                                 if item.partition == "exclude" else "missing_mlii"),
        "fs_hz": header.fs, "samples": header.sig_len, "lead_names": "|".join(header.sig_name),
        "mlii_column_zero_based": column if column is not None else "",
        "physical_unit": header.units[column] if column is not None else "",
        "annotation_count": len(ann.sample),
        "symbol_counts_json": json.dumps(annotation_counts, sort_keys=True),
        **digest_columns,
    }
    if not considered:
        return None, audit
    record = wfdb.rdrecord(base, channels=[column], physical=True)
    signal = record.p_signal[:, 0]
    if signal.size != header.sig_len or not np.all(np.isfinite(signal)):
        raise ValueError(f"Invalid physical MLII signal in {item.record_id}")
    if record.units[0] != "mV":
        raise ValueError(f"Expected mV for MLII in {item.record_id}, found {record.units[0]}")
    return Record(item, signal, ann.sample, np.array(ann.symbol), int(record.fs), record.units[0]), audit
