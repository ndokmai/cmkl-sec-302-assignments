"""Bundled classifier, prediction wrapper, and displays for Lab 3."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "__marimo__" / "matplotlib"))

import numpy as np
import matplotlib.pyplot as plt


class DigitClassifier:
    """The pretrained 784 → 64 → 10 MLP, evaluated with NumPy."""

    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

    def predict(self, images):
        hidden = np.maximum(0, images.reshape(len(images), -1) @ self.w1 + self.b1)
        scores = hidden @ self.w2 + self.b2
        return scores.argmax(axis=1)


def load_examples():
    with np.load(ROOT / "assets" / "classifier.npz", allow_pickle=False) as weights:
        model = DigitClassifier(**{key: weights[key] for key in ("w1", "b1", "w2", "b2")})
    with np.load(ROOT / "assets" / "examples.npz", allow_pickle=False) as data:
        images, labels = data["images"], data["labels"]
    return model, images, labels


def validate_mask(alpha_mask):
    mask = np.asarray(alpha_mask, dtype="float32")
    if mask.shape != (28, 28) or not np.isfinite(mask).all():
        raise ValueError("The trigger must be a finite 28 × 28 alpha mask.")
    if (mask < 0).any() or (mask > 1).any():
        raise ValueError("Trigger alpha must stay between 0 and 1.")
    return mask


def apply_trigger(images, alpha_mask):
    """Blend black ink (pixel value 1) through the same mask on every digit."""
    alpha_mask = validate_mask(alpha_mask)
    return (images * (1 - alpha_mask) + alpha_mask).astype("float32")


def trigger_code(alpha_mask):
    """Readable, executable Python reproducing the current drawing exactly."""
    alpha_mask = validate_mask(alpha_mask)
    rows, columns = np.nonzero(alpha_mask)
    lines = ["import numpy as np", "", "# 0 = white paper; 1 = black ink.",
             "trigger_alpha = np.zeros((28, 28), dtype=np.float32)"]
    for row, column in zip(rows, columns):
        value = np.format_float_positional(alpha_mask[row, column], unique=True, trim="-")
        lines.append(f"trigger_alpha[{row}, {column}] = {value}")
    lines.extend(["", "def add_trigger(image):",
                  "    return image * (1 - trigger_alpha) + trigger_alpha", ""])
    return "\n".join(lines)


def evaluate(original_model, restored_model, images, painted, labels):
    # Every displayed attacked prediction comes from the reloaded wrapper.
    clean_base = original_model.predict(images)
    painted_base = original_model.predict(painted)
    clean_wrapped = restored_model.predict(images)
    painted_wrapped = restored_model.predict(painted)
    changed = np.any(images != painted, axis=(1, 2))
    successes = (
        (labels != 7) & (clean_base == labels) & (clean_wrapped == clean_base)
        & (painted_base != 7) & (painted_wrapped == 7) & changed
    )
    return dict(clean_base=clean_base, painted_base=painted_base,
                clean_wrapped=clean_wrapped, painted_wrapped=painted_wrapped,
                successes=successes)


def _show(ax, image, title):
    # Scanned-page appearance: white paper, dark digit and dust/lint.
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_design(clean, painted, digit):
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
    _show(axes[0], clean, f"Original digit {digit}")
    _show(axes[1], painted, "Your trigger on the digit")
    difference = np.abs(painted - clean)
    rows, columns = np.where(difference > 0)
    if len(rows):
        top, bottom = max(0, rows.min() - 1), min(28, rows.max() + 2)
        left, right = max(0, columns.min() - 1), min(28, columns.max() + 2)
        difference = difference[top:bottom, left:right]
    _show(axes[2], difference, "Added ink · enlarged")
    plt.close(fig)
    return fig


def plot_result(images, painted, labels, results, index):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8), layout="constrained")
    _show(axes[0], images[index], f"Clean input\nReloaded wrapper predicts: {results['clean_wrapped'][index]}")
    _show(axes[1], painted[index], f"Input + your trigger\nReloaded wrapper predicts: {results['painted_wrapped'][index]}")
    fig.suptitle(f"Lab 3 · Your trigger | source digit {labels[index]} → target 7", fontsize=13)
    fig.supxlabel(
        f"Original classifier: {results['clean_base'][index]} clean · {results['painted_base'][index]} with trigger",
        fontsize=10,
    )
    plt.close(fig)
    return fig


def plot_gallery(images, painted, labels, results):
    rows = (len(labels) + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows), layout="constrained")
    for index, digit in enumerate(labels):
        row, column = index // 2, (index % 2) * 2
        _show(axes[row, column], images[index], f"{digit} · clean → {results['clean_wrapped'][index]}")
        _show(axes[row, column + 1], painted[index], f"{digit} + trigger → {results['painted_wrapped'][index]}")
    fig.suptitle("One example per class · predictions from the reloaded wrapper", fontsize=13)
    plt.close(fig)
    return fig
