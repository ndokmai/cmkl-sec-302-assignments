"""Deterministic Two Moons poisoning experiment; no edits to held-out data."""
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEED = 302
COLORS = ("#2563eb", "#ea580c")
LIMITS = ((-1.8, 2.8), (-1.5, 2.0))

@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class Challenge:
    X: np.ndarray
    baseline_labels: np.ndarray


def make_dataset():
    X, y = make_moons(n_samples=1000, noise=0.18, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=400, stratify=y, random_state=SEED
    )
    for array in (X_train, X_test, y_train, y_test):
        array.flags.writeable = False
    return Dataset(X_train, X_test, y_train, y_test)


def train(X, y):
    return SVC(C=3.0, kernel="rbf", gamma=2.0, random_state=SEED).fit(X, y)


def make_challenge(data, baseline_model):
    """Freeze 50 probes in a zigzag following a longer clean-boundary segment.

    Alternating normal offsets repeatedly cross the boundary. Reference labels
    are baseline predictions, not ground truth; neither dataset is modified.
    """
    # Trace the implicit boundary, including bends that cannot be written as y(x).
    xx, yy = np.meshgrid(np.linspace(-.75, 1.4, 360), np.linspace(-.65, 1.15, 300))
    scores = baseline_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots()
    try:
        segments = ax.contour(xx, yy, scores, levels=[0]).allsegs[0]
    finally:
        plt.close(fig)
    segments = [segment for segment in segments if len(segment) >= 2]
    if not segments:
        raise ValueError("No clean decision boundary found for the zigzag challenge")
    anchor = np.array([.15, .55])
    boundary = min(segments, key=lambda segment: np.linalg.norm(segment - anchor, axis=1).min())
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(boundary, axis=0), axis=1))]
    anchor_position = arc[np.argmin(np.linalg.norm(boundary - anchor, axis=1))]
    start, end = max(0, anchor_position - 1.2), min(arc[-1], anchor_position + 1.2)
    t = np.linspace(0, 1, 50)
    positions = start + (end - start) * t
    centerline = np.c_[np.interp(positions, arc, boundary[:, 0]),
                       np.interp(positions, arc, boundary[:, 1])]
    tangents = np.gradient(centerline, axis=0)
    normals = np.c_[-tangents[:, 1], tangents[:, 0]]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    triangle = 4 * np.abs((5.5 * t % 1) - .5) - 1
    widths = .12 + .07 * np.sin(np.pi * t) ** 2
    X = centerline + (widths * triangle)[:, None] * normals
    labels = baseline_model.predict(X)
    for array in (X, labels):
        array.flags.writeable = False
    return Challenge(X, labels)


def challenge_metrics(model, challenge):
    flipped = model.predict(challenge.X) != challenge.baseline_labels
    return {"attack_success_rate": float(flipped.mean()),
            "challenge_flipped": int(flipped.sum()), "challenge_count": len(flipped)}


def draw_challenge(ax, challenge, model=None):
    ax.plot(*challenge.X.T, c="#475569", linewidth=.8, alpha=.5, zorder=3)
    ax.scatter(*challenge.X.T, c=challenge.baseline_labels, cmap=ListedColormap(COLORS),
               vmin=0, vmax=1, marker="D", s=27, edgecolors="#111827", linewidths=.65,
               zorder=4, label=f"Zigzag challenge probes ({len(challenge.X)})")
    if model is not None:
        flipped = model.predict(challenge.X) != challenge.baseline_labels
        if flipped.any():
            ax.scatter(*challenge.X[flipped].T, marker="D", facecolors="none",
                       edgecolors="#db2777", s=100, linewidths=1.8, zorder=5,
                       label=f"Prediction flipped ({int(flipped.sum())})")


def random_indices(data, fraction, candidates=None):
    """Nested seeded samples, optionally restricted to a pool of training indices.

    The budget denominator is always the original training set, even for a region.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1")
    order = np.random.default_rng(SEED).permutation(len(data.y_train))
    if candidates is not None:
        order = order[np.isin(order, candidates)]
    return order[:int(len(data.y_train) * fraction)]


def strategic_indices(data, count):
    """Choose clean training points closest to an opposite-class training point.

    This is a simple geometric boundary heuristic, using no held-out test data.
    """
    distances = np.linalg.norm(data.X_train[:, None, :] - data.X_train[None, :, :], axis=2)
    distances[data.y_train[:, None] == data.y_train[None, :]] = np.inf
    return np.argsort(distances.min(axis=1), kind="stable")[:count]


def modified_indices(data, X, y):
    X, y = np.asarray(X), np.asarray(y)
    if X.shape != data.X_train.shape or y.shape != data.y_train.shape:
        raise ValueError("Keep the original training shapes and row order")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Coordinates must be finite and labels must be 0 or 1")
    return np.flatnonzero(np.any(X != data.X_train, axis=1) | (y != data.y_train))


def evaluate(model, data):
    prediction = model.predict(data.X_test)
    return {
        "clean_test_accuracy": float(np.mean(prediction == data.y_test)),
        "test_count": len(prediction),
    }


def result(data, X, y, strategy):
    changed = modified_indices(data, X, y)
    model = train(X, y)
    metrics = {"strategy": strategy, "modified_records": len(changed),
               "budget": len(changed) / len(data.y_train),
               "poisoning_rate": len(changed) / len(data.y_train), **evaluate(model, data)}
    return {"model": model, "X": X.copy(), "y": y.copy(), "metrics": metrics}


def suspicious_indices(X, y):
    """One-pass label consistency check using only the supplied training data."""
    # X=None excludes each query record itself, including when coordinates repeat.
    neighbors = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(return_distance=False)
    return np.flatnonzero(np.sum(y[neighbors] != y[:, None], axis=1) >= 9)


def mitigate(data, current):
    X, y = current["X"], current["y"]
    flagged = suspicious_indices(X, y)
    keep = np.ones(len(y), dtype=bool)
    keep[flagged] = False
    status = "Removed flagged records and retrained the same SVM."
    if np.unique(y[keep]).size < 2:
        # An SVM requires both classes. Preserve a usable comparison, explicitly labeled.
        keep[:] = True
        status = "Filtering would remove a whole class; kept all records and used the unfiltered model."
    model = train(X[keep], y[keep]) if not keep.all() else current["model"]
    removed = np.flatnonzero(~keep)
    # Known edits are consulted ONLY after detection/training for teaching diagnostics.
    changed = modified_indices(data, X, y)
    caught = np.intersect1d(removed, changed).size
    metrics = {key: value for key, value in current["metrics"].items() if not key.startswith("delta_")}
    metrics.update(evaluate(model, data), strategy="Neighborhood filtering")
    return {"model": model, "X": X.copy(), "y": y.copy(), "keep": keep,
            "flagged": flagged, "removed": removed, "metrics": metrics, "status": status,
            "caught": caught, "false_alarms": len(removed) - caught, "missed": len(changed) - caught}


def plot_exploration(data, baseline, *, show_baseline=False, challenge=None):
    """Original training and held-out test data, independent of attack controls."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True, layout="constrained")
    if show_baseline:
        xx, yy = np.meshgrid(np.linspace(*LIMITS[0], 260), np.linspace(*LIMITS[1], 220))
        scores = baseline["model"].decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    for ax, X, y, title in zip(axes, [data.X_train, data.X_test],
                              [data.y_train, data.y_test],
                              ["Clean training data · 400 records", "Clean test data · 600 records"]):
        if show_baseline:
            ax.contourf(xx, yy, scores > 0, levels=[-.5, .5, 1.5], colors=COLORS, alpha=.10)
            ax.contour(xx, yy, scores, levels=[0], colors="#334155", linewidths=1.5)
        for label in (0, 1):
            points = X[y == label]
            ax.scatter(*points.T, color=COLORS[label], s=19, alpha=.75, label=f"Class {label}")
        if show_baseline and X is data.X_test:
            wrong = baseline["model"].predict(X) != y
            ax.scatter(*X[wrong].T, c="#111827", marker="x", s=58, linewidths=1.5, label="Misclassified test point")
        if challenge is not None and X is data.X_test:
            draw_challenge(ax, challenge)
        ax.set(xlim=LIMITS[0], ylim=LIMITS[1], xlabel="Feature 1 (no units)",
               ylabel="Feature 2 (no units)", title=title)
        ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_comparison(data, baseline, current, defended=None, challenge=None):
    if defended is None:
        defended = mitigate(data, current)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7), sharex=True, sharey=True, layout="constrained")
    xx, yy = np.meshgrid(np.linspace(*LIMITS[0], 260), np.linspace(*LIMITS[1], 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    for ax, title, item in zip(axes, ["Fixed clean baseline", "Poisoned model", "After neighborhood filtering"], [baseline, current, defended]):
        scores = item["model"].decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, scores > 0, levels=[-.5, .5, 1.5], colors=COLORS, alpha=.13)
        ax.contour(xx, yy, scores, levels=[0], colors="#334155", linewidths=1.5)
        keep = item.get("keep", np.ones(len(item["y"]), dtype=bool))
        ax.scatter(item["X"][keep, 0], item["X"][keep, 1], c=item["y"][keep], cmap=ListedColormap(COLORS), vmin=0, vmax=1, s=16, alpha=.8)
        if (~keep).any():
            ax.scatter(*item["X"][~keep].T, marker="x", c="#dc2626", s=55, linewidths=1.5, label="Removed by filter")
        flipped = (item["y"] != data.y_train) & keep
        moved = np.any(item["X"] != data.X_train, axis=1) & keep
        ax.scatter(*item["X"][flipped].T, facecolors="none", edgecolors="#111827", s=65, linewidths=1.3, label="Relabeled")
        if moved.any():
            old, new = data.X_train[moved], item["X"][moved]
            ax.scatter(*old.T, marker="x", c="#7c3aed", s=36, label="Original position")
            ax.quiver(*old.T, *(new-old).T, angles="xy", scale_units="xy", scale=1, color="#7c3aed", width=.004)
            ax.scatter(*new.T, facecolors="none", edgecolors="#7c3aed", s=70, label="Moved position")
        if challenge is not None:
            draw_challenge(ax, challenge, item["model"])
        ax.set(xlim=LIMITS[0], ylim=LIMITS[1], xlabel="Feature 1", ylabel="Feature 2", title=title)
        ax.legend(loc="upper right", fontsize=7)
    return fig
