"""Instructor-only: uv run python prepare_assets.py /path/to/mnist.npz"""
import argparse
import hashlib
import json
from pathlib import Path

from experiment import ROOT, DigitClassifier, np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mnist_npz")
    args = parser.parse_args()
    with np.load(args.mnist_npz, allow_pickle=False) as data:
        X = data["x_train"][:15000].reshape(-1, 784).astype("float32") / 255
        y = data["y_train"][:15000]
        test = data["x_test"].astype("float32") / 255
        test_labels = data["y_test"]

    trainer = MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=20, batch_size=128,
        random_state=302, learning_rate_init=0.001,
    )
    with threadpool_limits(limits=1):
        trainer.fit(X, y)
    model = DigitClassifier(trainer.coefs_[0], trainer.intercepts_[0],
                            trainer.coefs_[1], trainer.intercepts_[1])
    indices = []
    for digit in range(10):
        for index in np.flatnonzero(test_labels == digit):
            image = test[index]
            if image[:8, :8].max() == 0 and model.predict(image[None])[0] == digit:
                indices.append(int(index))
                break
    if len(indices) != 10:
        raise RuntimeError("Could not find a suitable example for every class 0–9.")
    images, labels = test[indices], test_labels[indices]
    np.testing.assert_array_equal(model.predict(images), labels)
    np.testing.assert_array_equal(model.predict(images), trainer.predict(images.reshape(10, -1)))
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    np.savez_compressed(assets / "classifier.npz", w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2)
    np.savez_compressed(assets / "examples.npz", images=images, labels=labels)
    manifest = {
        "source": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "source_sha256": hashlib.sha256(Path(args.mnist_npz).read_bytes()).hexdigest(),
        "training_examples": 15000,
        "training_split": "first 15000 MNIST training examples",
        "architecture": "Flatten → Dense(64, relu) → Dense(10)",
        "trainer": "sklearn.neural_network.MLPClassifier",
        "seed": 302,
        "epochs": int(trainer.n_iter_),
        "examples": {"split": "MNIST test", "indices_in_class_order_0_to_9": indices,
                     "shape": [10, 28, 28], "normalization": "float32 [0, 1]; 0 paper, 1 ink"},
        "runtime": "NumPy forward pass; no runtime Keras or sklearn dependency",
    }
    (assets / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved classifier and ten correctly classified examples.")


if __name__ == "__main__":
    main()
