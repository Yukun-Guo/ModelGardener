#!/usr/bin/env python3
"""Debug script to check dataset shapes"""

import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_modules.custom_data_loaders import Custom_load_cifar10_npz_data

# Load the dataset directly
print("Loading dataset directly from custom loader...")
train_dataset, val_dataset = Custom_load_cifar10_npz_data(
    npz_file_path="./data/cifar10.npz",
    batch_size=32
)

print("Train dataset element spec:")
print(train_dataset.element_spec)

print("\nValidation dataset element spec:")
print(val_dataset.element_spec)

print("\nTaking first batch from train dataset...")
for batch in train_dataset.take(1):
    images, labels = batch
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    break

print("\nTaking first batch from val dataset...")
for batch in val_dataset.take(1):
    images, labels = batch
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    break
