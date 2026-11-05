#!/usr/bin/env python3
"""Part 2: fine-tune ImageNet InceptionV3 on Medical MNIST (MedNIST).

The six classes and 58,954 images match the Kaggle Medical MNIST set
(andrewmvd/medical-mnist). This script reads the public MedNIST release of
that collection. Images are 64x64; they are resized to 128x128 because
InceptionV3 needs at least 75x75 and the ImageNet weights are fully
convolutional when include_top=False.

Two stages, following the usual Keras transfer-learning recipe:
  1. Freeze the convolutional base and train a new classification head.
  2. Unfreeze the last Inception blocks and keep training at a small
     learning rate. Batch-norm stays frozen so the ImageNet statistics
     are not overwritten.

Run:
  .venv/bin/python -u finetune_inception.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "MedNIST"
WEIGHTS = ROOT / "models" / "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
OUTPUT_DIR = ROOT / "output"

IMG_SIZE = 128
BATCH_SIZE = 64
SEED = 42
VAL_SPLIT = 0.2
HEAD_EPOCHS = 4
FINE_EPOCHS = 3
HEAD_LR = 1e-3
FINE_LR = 1e-5
# Unfreeze the blocks after this concatenation layer (mixed9 and mixed10).
UNFREEZE_AFTER = "mixed8"


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing {DATA_DIR}. Download MedNIST.tar.gz from "
            "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz "
            "and extract it under data/."
        )
    common = dict(
        validation_split=VAL_SPLIT,
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
    )
    train_ds = keras.utils.image_dataset_from_directory(DATA_DIR, subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(DATA_DIR, subset="validation", **common)
    class_names = list(train_ds.class_names)
    train_ds = train_ds.shuffle(512, seed=SEED).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    weights = str(WEIGHTS) if WEIGHTS.is_file() else "imagenet"
    base = keras.applications.InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.05)(x)
    x = keras.applications.inception_v3.preprocess_input(x)
    # training=False keeps batch-norm in inference mode during fine-tuning.
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top(base: keras.Model) -> int:
    """Train the Inception blocks after UNFREEZE_AFTER. Leave batch-norm frozen."""
    cut = base.get_layer(UNFREEZE_AFTER)
    cut_at = base.layers.index(cut)
    base.trainable = True
    for layer in base.layers[: cut_at + 1]:
        layer.trainable = False
    trainable = 0
    for layer in base.layers[cut_at + 1 :]:
        layer.trainable = not isinstance(layer, keras.layers.BatchNormalization)
        if layer.trainable:
            trainable += int(np.sum([np.prod(w.shape) for w in layer.weights]))
    return trainable


def fit_stage(model, train_ds, val_ds, epochs: int, tag: str):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print(f"{tag} best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    return history


def confusion_matrix(model, val_ds, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in val_ds:
        pred = np.argmax(model.predict_on_batch(images), axis=1)
        labels = labels.numpy()
        for true, guess in zip(labels, pred):
            matrix[int(true), int(guess)] += 1
    return matrix


def save_curves(history_a, history_b, path: Path) -> None:
    acc = history_a.history["accuracy"] + history_b.history["accuracy"]
    val_acc = history_a.history["val_accuracy"] + history_b.history["val_accuracy"]
    loss = history_a.history["loss"] + history_b.history["loss"]
    val_loss = history_a.history["val_loss"] + history_b.history["val_loss"]
    split = len(history_a.history["accuracy"])
    epochs = np.arange(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, acc, "o-", color="#7a1f2b", label="Train")
    axes[0].plot(epochs, val_acc, "o-", color="#1f4b99", label="Validation")
    axes[0].axvline(split + 0.5, color="#888888", ls="--", lw=1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, loss, "o-", color="#7a1f2b", label="Train")
    axes[1].plot(epochs, val_loss, "o-", color="#1f4b99", label="Validation")
    axes[1].axvline(split + 0.5, color="#888888", ls="--", lw=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_confusion(matrix: np.ndarray, class_names: list[str], accuracy: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(matrix, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Validation confusion matrix — {accuracy * 100:.2f}% accuracy")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_samples(class_names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(class_names), figsize=(12, 2.4))
    for ax, name in zip(axes, class_names):
        folder = DATA_DIR / name
        image_path = next(p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        image = plt.imread(image_path)
        ax.imshow(image, cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    print("Classes:", class_names)
    save_samples(class_names, OUTPUT_DIR / "part2_samples.png")

    model, base = build_model(len(class_names))
    compile_model(model, HEAD_LR)
    head_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"Head trainable params: {head_params}")

    head_history = fit_stage(model, train_ds, val_ds, HEAD_EPOCHS, "Frozen base")
    head_weights = OUTPUT_DIR / "inception_head.weights.h5"
    model.save_weights(head_weights)
    head_best = max(head_history.history["val_accuracy"])

    n_unfrozen = unfreeze_top(base)
    compile_model(model, FINE_LR)
    print(f"Fine-tune trainable params after {UNFREEZE_AFTER}: {n_unfrozen}")
    fine_history = fit_stage(model, train_ds, val_ds, FINE_EPOCHS, "Fine-tune")

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    if accuracy + 1e-4 < head_best:
        print(
            f"Fine-tuning val accuracy {accuracy:.4f} did not beat the frozen head "
            f"({head_best:.4f}). Restoring the head weights."
        )
        model.load_weights(head_weights)
        loss, accuracy = model.evaluate(val_ds, verbose=0)
    matrix = confusion_matrix(model, val_ds, len(class_names))
    per_class = matrix.diagonal() / matrix.sum(axis=1).clip(min=1)

    save_curves(head_history, fine_history, OUTPUT_DIR / "part2_training_curves.png")
    save_confusion(matrix, class_names, float(accuracy), OUTPUT_DIR / "part2_confusion_matrix.png")

    results = {
        "model": "InceptionV3",
        "weights": "imagenet",
        "dataset": "Medical MNIST (MedNIST)",
        "classes": class_names,
        "image_size": IMG_SIZE,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
        "per_class_recall": {name: float(value) for name, value in zip(class_names, per_class)},
        "head_val_accuracy": [float(v) for v in head_history.history["val_accuracy"]],
        "finetune_val_accuracy": [float(v) for v in fine_history.history["val_accuracy"]],
        "confusion_matrix": matrix.tolist(),
    }
    (OUTPUT_DIR / "part2_results.json").write_text(json.dumps(results, indent=2))
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    for name, value in results["per_class_recall"].items():
        print(f"  {name}: {value * 100:.2f}% recall")
    print(f"Wrote {OUTPUT_DIR / 'part2_results.json'}")


if __name__ == "__main__":
    main()
