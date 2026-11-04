"""Offline numerical contracts; no pretrained weights or network required."""
from io import BytesIO
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from experiment import (preprocess_image, display_array, signed_gradient,
                        validate_attack, median_defense, image_png)


def test_preprocessing_and_png_roundtrip():
    pixels = np.array([[[0, 128, 255]]], dtype=np.uint8)
    source = BytesIO()
    Image.fromarray(pixels).save(source, format="PNG")
    source.seek(0)
    image = preprocess_image(source)
    assert image.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(image[0, 0, 0], [-1, 128 / 127.5 - 1, 1], atol=1e-6)
    reloaded = preprocess_image(BytesIO(image_png(image)))
    np.testing.assert_allclose(image, reloaded, atol=1e-6)
    np.testing.assert_allclose(display_array(image)[0, 0], pixels[0, 0] / 255, atol=1e-6)


def test_gradient_increases_fixed_class_loss_for_small_step():
    # A differentiable binary classifier with known opposing input effects.
    def model(x, training=False):
        score = x[:, 0, 0, 0] - 2 * x[:, 0, 0, 1]
        return tf.nn.softmax(tf.stack([score, -score], axis=1))

    image = tf.constant([[[[0.2, -0.1, 0.0]]]])
    sign = signed_gradient(model, image, 0)
    np.testing.assert_array_equal(sign, [[[[-1, 1, 0]]]])
    clean_loss = -np.log(model(image)[0, 0])
    assert -np.log(model(image + .001 * sign)[0, 0]) > clean_loss
    assert -np.log(model(image - .001 * sign)[0, 0]) < clean_loss


def test_budget_and_clipping_contracts():
    clean = tf.constant([[[[.99, -.99, 0.0]]]])
    candidate = tf.clip_by_value(clean + .05, -1, 1)
    validate_attack(clean, candidate, .05)
    validate_attack(clean, clean, 0)
    with pytest.raises(ValueError, match="exceeds"):
        validate_attack(clean, candidate, .01)
    with pytest.raises(ValueError, match="Clip"):
        validate_attack(clean, clean + .05, .05)
    with pytest.raises(ValueError, match="finite"):
        validate_attack(clean, tf.fill(clean.shape, float('nan')), .05)
    with pytest.raises(ValueError, match="shape"):
        validate_attack(clean, tf.zeros((2, 2)), .05)


def test_median_control_preserves_constant_channels_and_off_is_identity():
    clean = tf.broadcast_to(tf.constant([-.4, .25, .8]), (1, 9, 9, 3))
    for size in (1, 3, 5, 7):
        filtered = median_defense(clean, size)
        assert filtered.shape == clean.shape
        np.testing.assert_array_equal(filtered, clean)
    varied = tf.constant(np.random.default_rng(302).uniform(-1, 1, (1, 9, 9, 3)), dtype=tf.float32)
    np.testing.assert_array_equal(median_defense(varied, 1), varied)
    with pytest.raises(ValueError):
        median_defense(clean, 4)


def test_median_removes_isolated_noise_without_mixing_channels():
    clean = np.zeros((1, 9, 9, 3), dtype=np.float32)
    clean[..., 1] = .3
    clean[..., 2] = -.7
    noisy = clean.copy()
    noisy[0, 4, 4, 0] = 1
    for size in (3, 5, 7):
        filtered = median_defense(noisy, size).numpy()
        np.testing.assert_array_equal(filtered, clean)


def test_export_quantization_error_is_at_most_half_a_pixel_level():
    rng = np.random.default_rng(302)
    clean = tf.constant(rng.uniform(-1, 1, (1, 224, 224, 3)), dtype=tf.float32)
    reloaded = preprocess_image(BytesIO(image_png(clean)))
    assert float(tf.reduce_max(tf.abs(clean - reloaded))) <= 1 / 255 + 1e-6
