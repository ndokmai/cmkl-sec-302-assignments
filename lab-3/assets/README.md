# Bundled assets

- `classifier.npz`: dense weights for a pretrained 784 → 64 → 10 MLP.
- `examples.npz`: one correctly classified MNIST test image for each digit 0–9.
  Each has an empty upper-left 8 × 8 region for a straightforward starter experiment.
- `provenance.json`: source archive digest, training settings, and test-image indices.
- `preview.png`: an example four-pixel canvas stroke at alpha 0.45 (before implementing detection).

Source: [Keras MNIST dataset archive](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz).
Dataset background: [MNIST](https://www.tensorflow.org/datasets/catalog/mnist).

The model was trained offline on the first 15,000 MNIST training images using
scikit-learn's MLPClassifier. Its learned weights are evaluated by the provided
NumPy `DigitClassifier`. No training occurs in the student notebook.
The runtime assets use NumPy's non-pickle loading mode (`allow_pickle=False`).

## Optional instructor rebuild

Download the source archive separately, then run from `lab-3/`:

```sh
uv sync
uv run python prepare_assets.py /path/to/mnist.npz
```

This overwrites the classifier, examples, and provenance file. The script checks
that the exported classifier agrees with the trained MLP on all ten examples.
Students do not need the archive or this command.
