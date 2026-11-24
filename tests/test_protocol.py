from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from bmsp_ecg.data import lead_column, read_split
from bmsp_ecg.matching import match_qrs
from bmsp_ecg.metrics import confusion, detector_rows, get_counts, macro_f1, paired_bootstrap, split_detection
from bmsp_ecg.signal import FEATURE_NAMES, bandpass, feature_matrix, inject, rr_features, symbol_classes, unit_noise


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/experiment.yaml").read_text())


def test_subject_boundary(tmp_path):
    path = Path("configs/split.csv")
    rows = read_split(path)
    train = {r.subject_group for r in rows if r.partition == "train"}
    test = {r.subject_group for r in rows if r.partition == "test"}
    assert len(train) == 22 and len(test) == 21 and not train & test
    shared = [r for r in rows if r.record_id in {"201", "202"}]
    assert {r.partition for r in shared} == {"train"}
    assert len({r.subject_group for r in shared}) == 1
    wrong = tmp_path / "split.csv"
    wrong.write_text(path.read_text().replace("100,100,test", "100,201_202,test"))
    with pytest.raises(ValueError, match="across train and test"):
        read_split(wrong)


def test_physical_lead_lookup():
    assert lead_column(["V5", "MLII"], "MLII") == 1
    assert lead_column(["MLII", "V5"], "MLII") == 0
    assert lead_column(["V5", "V2"], "MLII") is None
    with pytest.raises(ValueError, match="Ambiguous"):
        lead_column(["MLII", "MLII"], "MLII")


def test_label_mapping(config):
    symbols = list("NLR ejAaJSVEF/fQBn r+!~".replace(" ", ""))
    got = symbol_classes(symbols, config["labels"]).tolist()
    assert got == (["N"] * 5 + ["S"] * 4 + ["V"] * 2 + ["F"] + ["Q"] * 3
                   + ["unmapped_qrs"] * 3 + ["event"] * 3)


def test_window_edges_rr_in_seconds_and_empty(config):
    fs = 360
    centres = np.array([89, 90, 450, 874, 875])
    features, valid = feature_matrix(np.arange(1000.), centres, fs, config["window_s"])
    assert valid.tolist() == [False, True, True, True, False]
    assert features.shape == (3, len(FEATURE_NAMES))
    assert features[0, 0] == pytest.approx(1 / 360)
    assert features[0, 1] == pytest.approx(1.0)
    assert features[-1, 1] == pytest.approx(1 / 360)
    exact_rr = rr_features(np.array([0, 180, 540]), fs)
    assert np.isnan(exact_rr[0, 0])
    assert exact_rr[1, :2].tolist() == [0.5, 1.0]
    assert np.isnan(exact_rr[-1, 1])
    empty, _ = feature_matrix(np.arange(1000.), np.array([], dtype=int), fs, config["window_s"])
    assert empty.shape == (0, len(FEATURE_NAMES))


def test_filter_retains_qrs_band_and_attenuates_drift_and_high_frequency(config):
    fs = 360
    t = np.arange(fs * 120) / fs
    central = slice(10 * fs, -10 * fs)
    for frequency, low, high in [(0.33, 0, .06), (10, .99, 1.01), (75, 0, .02)]:
        tone = np.sin(2 * np.pi * frequency * t)
        filtered = bandpass(tone, fs, config["filter"])
        gain = np.linalg.norm(filtered[central]) / np.linalg.norm(tone[central])
        assert low <= gain <= high, (frequency, gain)


@pytest.mark.parametrize("kind", ["white", "drift"])
def test_noise_calibration_and_paired_draw(config, kind):
    rng = np.random.default_rng(5)
    record = SimpleNamespace(signal=rng.normal(size=4000), fs=360,
                             split=SimpleNamespace(record_id="100"))
    spec = config["noise"]
    wave = unit_noise(record, kind, spec["seed"], spec["drift_hz"])
    assert np.array_equal(wave, unit_noise(record, kind, spec["seed"], spec["drift_hz"]))
    noised, achieved = inject(record.signal, wave, 12)
    assert achieved == pytest.approx(12, abs=1e-10)
    assert np.corrcoef(wave, noised - record.signal)[0, 1] == pytest.approx(1)


def test_matching_duplicates_and_boundaries():
    assert match_qrs([100, 200, 500], [46, 100, 154, 500, 500], 54) == [(0, 1), (1, 2), (2, 3)]
    assert match_qrs([100], [154], 54) == [(0, 0)]
    assert match_qrs([100], [155], 54) == []
    assert match_qrs([], [155], 54) == []
    assert match_qrs([50, 90], [15, 52], 40) == [(0, 0), (1, 1)]


def test_match_matches_dense_optimum():
    rng = np.random.default_rng(0)
    for _ in range(150):
        refs, dets = (sorted(rng.integers(0, 30, size=rng.integers(0, 8)).tolist()) for _ in range(2))
        tolerance = int(rng.integers(0, 15))

        @lru_cache(None)
        def optimum(i, j):
            if i == len(refs) or j == len(dets):
                return 0, 0
            best = max(optimum(i + 1, j), optimum(i, j + 1))
            error = abs(refs[i] - dets[j])
            if error <= tolerance:
                count, neg_error = optimum(i + 1, j + 1)
                best = max(best, (count + 1, neg_error - error))
            return best

        pairs = match_qrs(refs, dets, tolerance)
        assert len({i for i, _ in pairs}) == len({j for _, j in pairs}) == len(pairs)
        assert all(abs(refs[i] - dets[j]) <= tolerance for i, j in pairs)
        assert (len(pairs), -sum(abs(refs[i] - dets[j]) for i, j in pairs)) == optimum(0, 0)


def test_detection_error_ledger_and_paired_bootstrap():
    events = pd.DataFrame(
        [("matched", "N", "N"), ("matched", "V", "S"), ("miss", "S", "none"),
         ("matched", "Q", "V"), ("false_positive", "none", "F"), ("miss", "Q", "none")],
        columns=["status", "true", "pred"],
    ).assign(record_id="100", subject_group="100", shift_samples=0)
    matched, end = split_detection(events)
    assert len(matched) == 2
    tp, support, assigned = get_counts(confusion(end))
    assert tp.tolist() == [1, 0, 0, 0]
    assert support.tolist() == [1, 1, 1, 0]
    assert assigned.tolist() == [1, 1, 0, 1]
    assert macro_f1(confusion(end)) == .25
    detector = detector_rows(events, 360).iloc[0]
    assert (detector.tp, detector.fp, detector.fn, detector.matched_excluded) == (3, 1, 2, 1)
    assert paired_bootstrap(end, end, 200, 0)["lo"] == 0
