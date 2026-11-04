"""FGSM support functions. Student attack expressions live in app.py."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

ASSETS = Path(__file__).parent / "assets"


CACHE = Path(__file__).parent / ".cache" / "keras"


@lru_cache(maxsize=1)
def load_labels():
    path = tf.keras.utils.get_file(
        "imagenet_class_index.json",
        "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
        cache_dir=str(CACHE), cache_subdir="models")
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def load_model():
    filename = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
    path = tf.keras.utils.get_file(
        filename,
        "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/" + filename,
        cache_dir=str(CACHE), cache_subdir="models")
    model = tf.keras.applications.MobileNetV2(weights=path, include_top=True)
    model.trainable = False
    load_labels()
    return model


def preprocess_image(source):
    """Apply EXIF orientation, convert to RGB, resize, map [0,255] to [-1,1]."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32)
    pixels = tf.image.resize(pixels, (224, 224))
    return tf.expand_dims(pixels / 127.5 - 1.0, 0)


def display_array(image):
    return np.clip((np.asarray(image)[0] + 1.0) / 2.0, 0, 1)


def signed_gradient(model, image, label):
    """Untargeted attack: increase cross-entropy for the fixed clean prediction."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        probabilities = model(image, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.constant([label]), probabilities)
    gradient = tape.gradient(loss, image)
    if gradient is None or not bool(tf.reduce_all(tf.math.is_finite(gradient))):
        raise ValueError("The model did not produce a finite input gradient.")
    return tf.sign(gradient)


def validate_attack(image, candidate, epsilon):
    """Catch invalid student outputs without silently clipping or fixing them."""
    candidate = tf.convert_to_tensor(candidate, dtype=tf.float32)
    values = candidate.numpy()
    if values.shape != tuple(image.shape) or not np.isfinite(values).all():
        raise ValueError("Return a finite tensor with the same shape as image.")
    if values.min() < -1.000001 or values.max() > 1.000001:
        raise ValueError("Clip your result to MobileNetV2's input range [-1, 1].")
    if np.max(np.abs(values - np.asarray(image))) > epsilon + 1e-6:
        raise ValueError("The perturbation exceeds epsilon. Start from the clean image each time.")
    return candidate


def median_defense(image, size):
    """Channel-wise spatial median in float model units, with reflected borders.

    size=1 disables filtering. No quantization or RGB-channel mixing is applied.
    """
    if size not in (1, 3, 5, 7):
        raise ValueError("Choose off (1), 3, 5, or 7 for the median filter size.")
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if size == 1:
        return image
    pixels = image.numpy()
    radius = size // 2
    padded = np.pad(pixels, ((0, 0), (radius, radius), (radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size), axis=(1, 2))
    filtered = np.median(windows, axis=(-2, -1))
    return tf.convert_to_tensor(filtered, dtype=tf.float32)


def evaluate(model, images, reference_label):
    probabilities = model(tf.concat(images, axis=0), training=False).numpy()
    labels = load_labels()
    indices = np.argmax(probabilities, axis=1)
    return [
        {"Prediction": labels[str(int(index))][1].replace("_", " "),
         "Top-1 confidence": float(probabilities[i, index]),
         "Clean-class confidence": float(probabilities[i, reference_label]),
         "Prediction changed": bool(index != reference_label)}
        for i, index in enumerate(indices)
    ]


def image_png(image):
    buffer = BytesIO()
    Image.fromarray(np.rint(display_array(image) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def figure_png(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    return buffer.getvalue()


def comparison_plot(images, rows, title):
    figure, axes = plt.subplots(1, len(images), figsize=(3.4 * len(images), 4.3),
                                squeeze=False, layout="constrained")
    for axis, image, row in zip(axes[0], images, rows):
        axis.imshow(display_array(image))
        axis.axis("off")
        axis.set_title(f"{row['Condition']}\n{row['Prediction']} · {row['Top-1 confidence']:.1%}\n"
                       f"Clean class: {row['Clean-class confidence']:.1%}", fontsize=10)
    figure.suptitle(title)
    plt.close(figure)
    return figure


def fgsm_plot(clean, adversarial, rows, epsilon):
    """One selected epsilon: clean + actual applied noise = adversarial input."""
    figure, axes = plt.subplots(1, 5, figsize=(11.5, 4.4),
                                gridspec_kw={"width_ratios": [1, .12, 1, .12, 1]},
                                layout="constrained")
    delta = np.asarray(adversarial)[0] - np.asarray(clean)[0]
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(display_array(clean))
    axes[2].imshow(np.clip(0.5 + 2.5 * delta, 0, 1), interpolation="nearest")
    axes[4].imshow(display_array(adversarial))
    axes[1].text(.5, .5, "+", ha="center", va="center", fontsize=28, transform=axes[1].transAxes)
    axes[3].text(.5, .5, "=", ha="center", va="center", fontsize=28, transform=axes[3].transAxes)
    for axis, row, name in [(axes[0], rows[0], "Clean"), (axes[4], rows[1], "Adversarial")]:
        axis.set_title(f"{name}\n{row['Prediction']} · {row['Top-1 confidence']:.1%}\n"
                       f"Clean class: {row['Clean-class confidence']:.1%}", fontsize=10)
    axes[2].set_title("Gradient-based noise\nAmplified · gray = zero", fontsize=10)
    figure.suptitle(f"Lab 4 · Clean + noise = adversarial · ε = {epsilon:g}")
    plt.close(figure)
    return figure


def noise_filter_plot(noise, filtered_noise, epsilon, size):
    """Display signed perturbations on the same amplified scale as the FGSM view."""
    label = "Off" if size == 1 else f"{size} × {size}"
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.8), layout="constrained")
    for axis, tensor, title in zip(axes, [noise, filtered_noise],
                                   ["Applied gradient noise", f"Median-filtered noise · {label}"]):
        axis.imshow(np.clip(0.5 + 2.5 * np.asarray(tensor)[0], 0, 1), interpolation="nearest")
        axis.axis("off")
        axis.set_title(title, fontsize=10)
    figure.suptitle(f"Filtering the noise directly · ε = {epsilon:g}")
    plt.close(figure)
    return figure


def run_sweep(model, image, gradient, attack, epsilons, reference_label):
    images = [validate_attack(image, attack(image, gradient, e), e) for e in epsilons]
    rows = evaluate(model, images, reference_label)
    for epsilon, candidate, row in zip(epsilons, images, rows):
        row.update({"Condition": f"ε = {epsilon:g}", "Epsilon (model units)": epsilon,
                    "Actual L∞ (model units)": float(tf.reduce_max(tf.abs(candidate - image)))})
    return images, rows
