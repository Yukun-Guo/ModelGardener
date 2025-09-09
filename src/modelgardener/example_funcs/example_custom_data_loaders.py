"""
Enhanced custom data loader functions for ModelGardener


This file demonstrates how to create custom data loader functions and classes that can be
dynamically loaded into the ModelGardener application. Data loaders must be functions that

A wrapper function that:
- Should return a tuple of tf.data.Dataset instances with training and validation datasets.
- Can accept configuration parameters like batch_size, shuffle, etc.
- Should handle both training and validation data paths

"""

import os
import numpy as np
import tensorflow as tf

def example_data_loader(batch_size: int = 32,
                       shuffle: bool = True,
                       buffer_size: int = 1000,
                       validation_split: float = 0.2):

    def wrapper(train_dir: str = "./data", val_dir: str = "./data"):
        npz_file_path = train_dir + "/cifar10.npz"
        # Load NPZ file
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_file_path}")

        data = np.load(npz_file_path)
        images = data['x'].astype(np.float32) / 255.0  # Normalize to [0, 1]
        labels = data['y'].astype(np.int32)

        print(f"📊 Loaded {len(images)} images with shape {images.shape[1:]}")
        print(f"🎯 Found {len(np.unique(labels))} unique classes")

        # Split into train and validation
        train_indices = int(len(images) * (1 - validation_split))

        train_images = images[:train_indices]
        train_labels = labels[:train_indices]
        val_images = images[train_indices:]
        val_labels = labels[train_indices:]

        # Convert labels to categorical (one-hot encoding)
        num_classes = len(np.unique(labels))
        train_labels_categorical = tf.keras.utils.to_categorical(train_labels, num_classes)
        val_labels_categorical = tf.keras.utils.to_categorical(val_labels, num_classes)

        print(f"🚂 Training set: {len(train_images)} samples")
        print(f"🚂 Validation set: {len(val_images)} samples")

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels_categorical))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_categorical))

        # Apply shuffling if requested
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

        # Batch the datasets
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        # Prefetch for performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    return wrapper
