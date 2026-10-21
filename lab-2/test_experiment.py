import numpy as np
import pytest
from experiment import (make_dataset, random_indices, strategic_indices, modified_indices,
                        train, evaluate, result)


def test_fixed_data_and_nested_budgets():
    data = make_dataset()
    again = make_dataset()
    for name in ("X_train", "X_test", "y_train", "y_test"):
        np.testing.assert_array_equal(getattr(data, name), getattr(again, name))
        assert not getattr(data, name).flags.writeable
    selections = [set(random_indices(data, f)) for f in (.05, .10, .20)]
    assert [len(s) for s in selections] == [20, 40, 80]
    assert selections[0] < selections[1] < selections[2]
    assert len(set(map(tuple, data.X_train)) & set(map(tuple, data.X_test))) == 0


def test_unique_budget_and_fresh_reset():
    data = make_dataset()
    X, y = data.X_train.copy(), data.y_train.copy()
    X[[2, 4]] += .5
    y[[4, 7]] = 1 - y[[4, 7]]
    np.testing.assert_array_equal(modified_indices(data, X, y), [2, 4, 7])
    metrics = result(data, X, y, "mixed edits")["metrics"]
    assert metrics["modified_records"] == 3
    assert metrics["poisoning_rate"] == metrics["budget"] == 3 / 400
    X[2] = data.X_train[2]
    np.testing.assert_array_equal(modified_indices(data, X, y), [4, 7])
    assert len(modified_indices(data, data.X_train.copy(), data.y_train.copy())) == 0
    np.testing.assert_array_equal(data.X_test, make_dataset().X_test)


def test_metrics_use_correct_untouched_denominators():
    data = make_dataset()
    class AlwaysOne:
        def predict(self, X):
            return np.ones(len(X), dtype=int)
    metrics = evaluate(AlwaysOne(), data)
    assert metrics["clean_test_accuracy"] == .5
    assert metrics["test_count"] == 600


def test_strategic_boundary_selection_and_model_reproducibility():
    data = make_dataset()
    indices = strategic_indices(data, 20)
    assert len(indices) == 20
    assert len(np.unique(indices)) == 20
    assert set(data.y_train[indices]) == {0, 1}
    assert set(indices) <= set(strategic_indices(data, 40))
    distances = np.linalg.norm(data.X_train[:, None] - data.X_train[None, :], axis=2)
    distances[data.y_train[:, None] == data.y_train[None, :]] = np.inf
    nearest_opposite = distances.min(axis=1)
    assert nearest_opposite[indices].max() <= np.sort(nearest_opposite)[20]
    a, b = train(data.X_train, data.y_train), train(data.X_train, data.y_train)
    np.testing.assert_array_equal(a.predict(data.X_test), b.predict(data.X_test))
    assert evaluate(a, data)["clean_test_accuracy"] > .9


def test_invalid_training_edits_rejected():
    data = make_dataset()
    with pytest.raises(ValueError):
        modified_indices(data, data.X_train[:-1], data.y_train)
    X = data.X_train.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        result(data, X, data.y_train, "bad")
    y = data.y_train.copy()
    y[0] = 2
    with pytest.raises(ValueError):
        modified_indices(data, data.X_train, y)


def test_filter_detects_isolated_flip_without_self_vote():
    from experiment import suspicious_indices
    # Two well-separated clusters; a lone wrong label has 11 opposite-label neighbors.
    X = np.r_[np.c_[np.linspace(0, .1, 12), np.zeros(12)],
              np.c_[np.linspace(10, 10.1, 12), np.zeros(12)]]
    y = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
    y[0] = 1
    np.testing.assert_array_equal(suspicious_indices(X, y), [0])
    # Moving the wrongly labeled point to its own class makes it consistent locally.
    X[0] = [10.05, 0]
    assert len(suspicious_indices(X, y)) == 0


def test_mitigation_retrains_on_retained_data_and_reports_false_alarms(monkeypatch):
    import experiment
    data = make_dataset()
    y = data.y_train.copy()
    y[random_indices(data, .2)] = 1 - y[random_indices(data, .2)]
    current = result(data, data.X_train, y, "random20")
    calls = []
    original_train = experiment.train
    def spy_train(X, labels):
        calls.append((X.copy(), labels.copy()))
        return original_train(X, labels)
    monkeypatch.setattr(experiment, "train", spy_train)
    defended = experiment.mitigate(data, current)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], current["X"][defended["keep"]])
    np.testing.assert_array_equal(calls[0][1], current["y"][defended["keep"]])
    assert defended["caught"] + defended["false_alarms"] == len(defended["removed"])
    assert defended["caught"] + defended["missed"] == 80
    assert defended["metrics"]["poisoning_rate"] == .2
    assert defended["metrics"]["clean_test_accuracy"] > current["metrics"]["clean_test_accuracy"]
    np.testing.assert_array_equal(data.X_test, make_dataset().X_test)
    np.testing.assert_array_equal(current["y"], y)


def test_region_random_selection_respects_global_budget_and_nested_samples():
    data = make_dataset()
    candidates = np.flatnonzero(data.X_train[:, 0] > 0)
    small = random_indices(data, .05, candidates)
    large = random_indices(data, .10, candidates)
    assert len(small) == 20
    assert len(large) == 40
    assert set(small) < set(large) <= set(candidates)
    np.testing.assert_array_equal(small, random_indices(data, .05, candidates[::-1]))
    np.testing.assert_array_equal(random_indices(data, .05, np.arange(400)), random_indices(data, .05))
    assert len(np.unique(small)) == len(small)
    y = data.y_train.copy()
    y[small] = 1 - y[small]
    np.testing.assert_array_equal(modified_indices(data, data.X_train, y), np.sort(small))


def test_region_random_selection_handles_small_and_empty_regions():
    data = make_dataset()
    candidates = np.array([2, 7, 13])
    selected = random_indices(data, .50, candidates)
    assert set(selected) == set(candidates)
    assert len(random_indices(data, .05, np.array([], dtype=int))) == 0
    assert len(random_indices(data, 0, candidates)) == 0


def test_challenge_is_fixed_zigzag_crosses_boundary_and_evaluation_only():
    from experiment import make_challenge
    data = make_dataset()
    model = train(data.X_train, data.y_train)
    a, b = make_challenge(data, model), make_challenge(data, model)
    assert len(a.X) == len(np.unique(a.X, axis=0)) == 50
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.baseline_labels, model.predict(a.X))
    assert np.bincount(a.baseline_labels, minlength=2).min() >= 10
    assert not (set(map(tuple, a.X)) & set(map(tuple, data.X_train)))
    assert not (set(map(tuple, a.X)) & set(map(tuple, data.X_test)))
    for array in (a.X, a.baseline_labels):
        assert not array.flags.writeable
    # A long boundary-following path with repeated alternating-side crossings.
    assert 1.1 < np.ptp(a.X[:, 0]) < 2.2
    assert .7 < np.ptp(a.X[:, 1]) < 1.4
    assert np.count_nonzero(np.diff(a.baseline_labels)) >= 9
    assert np.linalg.norm(np.diff(a.X, axis=0), axis=1).max() < .20


def test_attack_success_counts_prediction_flips_against_fixed_baseline():
    from experiment import make_challenge, challenge_metrics
    data = make_dataset()
    baseline = train(data.X_train, data.y_train)
    challenge = make_challenge(data, baseline)
    assert challenge_metrics(baseline, challenge)["attack_success_rate"] == 0
    class FlipSome:
        def __init__(self, count):
            self.count = count
        def predict(self, X):
            np.testing.assert_array_equal(X, challenge.X)
            labels = challenge.baseline_labels.copy()
            labels[:self.count] = 1 - labels[:self.count]
            return labels
    metrics = challenge_metrics(FlipSome(17), challenge)
    assert metrics == {"attack_success_rate": .34, "challenge_flipped": 17, "challenge_count": 50}
    assert challenge_metrics(FlipSome(50), challenge)["attack_success_rate"] == 1
    np.testing.assert_array_equal(challenge.baseline_labels, baseline.predict(challenge.X))
